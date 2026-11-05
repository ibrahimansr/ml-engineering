# Foundations: Computer Networking for ML Engineers

This chapter is for readers who know the ML math well but have never had to learn how computers talk to each other. Networking does not stay hidden for long once you work at scale. Your first multi-node job will hang at `init_process_group`. Your all-reduce will run at a tenth of the advertised speed. Somebody will ask you whether the cluster is "3:1 oversubscribed". This chapter builds the vocabulary from zero. By the end you should be able to read the [network chapter](../network/README.md), [communication patterns](../network/comms.md), [network debug](../network/debug/README.md), [network benchmarks](../network/benchmarks/README.md) and the more advanced [collective communication internals](../network/collectives-internals.md).

The goal is to explain why things are the way they are, not only what to type. Most networking failures in ML clusters make sense once you know which layer they happen at.

All the arithmetic in this chapter was checked with `python3`. Where a number depends on hardware generation or vendor, the text says so.

## Why an ML engineer needs this

Training a large model is a distributed program. Every optimizer step, thousands of accelerators exchange gradients, parameters or activations. On a good day the network keeps up with the compute. On a bad day, the GPUs that cost the most in the whole system sit idle while bytes crawl across a misconfigured link.

Three kinds of problems show up over and over:

1. **Connectivity:** process A cannot reach process B at all. Typical causes are a wrong IP, a blocked port, a hostname that resolves to the wrong address, or the wrong network interface.
2. **Performance:** things work, but slowly. Typical causes are traffic going over TCP instead of RDMA, the wrong NIC, an oversubscribed fabric, or a misread spec sheet.
3. **Reliability:** things work and then stop. Typical causes are a flapping link, a congested switch, or a timeout.

Each section below gives you the concepts needed to reason about one or more of these.

## Layers: the mental model

Network software is built in layers. Each layer offers a service to the layer above and relies only on the layer below. When you send bytes over a TCP socket, you do not care whether they travel over copper, fiber or Wi-Fi. That indifference is the whole point of layering.

### OSI vs TCP/IP

Textbooks teach the 7-layer OSI model. The Internet actually runs on the 4- or 5-layer TCP/IP model. Here is how they map, and what each layer means for you:

| OSI layer | TCP/IP layer | Example | What it means for you |
|---|---|---|---|
| 7 Application | Application | HTTP, SSH, DNS, NCCL bootstrap, gRPC | Your program's protocol |
| 6 Presentation | (folded into app) | TLS, serialization | Rarely a separate concern |
| 5 Session | (folded into app) | | Ignore |
| 4 Transport | Transport | TCP, UDP | Ports, reliability, congestion control |
| 3 Network | Internet | IP (v4/v6) | Addresses, subnets, routing between networks |
| 2 Data link | Link | Ethernet, InfiniBand link layer | MAC addresses, switches, frames, MTU |
| 1 Physical | Link (physical) | Copper, fiber, transceivers, cables | Link speed, cable faults, optics |

People say "L2" and "L3" all the time, and they mean the OSI numbers: an "L2 switch" forwards Ethernet frames, an "L3 switch" or router forwards IP packets, and an "L4 load balancer" looks at TCP/UDP ports. Layers 5 and 6 almost never come up.

One important note for later: **InfiniBand is a complete stack of its own.** It has its own physical, link, network and transport layers and does not use Ethernet or (by default) IP. RDMA over Converged Ethernet (RoCE) takes the InfiniBand transport and runs it on top of Ethernet/IP. Keep this in mind when we get to HPC networking.

### Encapsulation: packets, frames, headers

Each layer wraps the data from the layer above with its own header. This is called encapsulation. Suppose your program sends 1000 bytes over TCP on Ethernet:

```
  application data (1000 B)
  +--------------------------------------------+
  |                payload                     |
  +--------------------------------------------+

  TCP segment: TCP header (20 B, often 32 B with options) + payload
  +--------+-----------------------------------+
  |  TCP   |                payload            |
  +--------+-----------------------------------+

  IP packet: IPv4 header (20 B) + TCP segment
  +------+--------+----------------------------+
  |  IP  |  TCP   |                payload     |
  +------+--------+----------------------------+

  Ethernet frame: Eth header (14 B) + IP packet + FCS checksum (4 B)
  +-----+------+--------+--------------+-----+
  | Eth |  IP  |  TCP   |   payload    | FCS |
  +-----+------+--------+--------------+-----+
```

Each layer wraps the one above in its own header:

![encapsulation through the layers](images/udp-encapsulation.png)

([source](https://commons.wikimedia.org/wiki/File:UDP_encapsulation.svg))

The names change by layer. TCP calls its unit a **segment**, IP a **packet**, and Ethernet a **frame**. In casual speech everything is a "packet".

Each header holds the information its layer needs:

- **Ethernet:** destination MAC, source MAC, EtherType (what is inside, for example IPv4).
- **IP:** source IP, destination IP, time-to-live (TTL), protocol (TCP=6, UDP=17).
- **TCP:** source port, destination port, sequence number, acknowledgment number, flags, window size.

On the receiving side the layers unwrap in reverse order. A switch only looks at the Ethernet header. A router looks at the IP header. Only the end host looks at TCP.

### MTU and jumbo frames

The **MTU** (Maximum Transmission Unit) is the largest IP packet a link will carry in one frame. Standard Ethernet MTU is **1500 bytes**. **Jumbo frames** are usually **9000 bytes**. Switches often allow a bit more, such as 9216, to leave room for extra headers.

Why does MTU matter? Every frame has a fixed cost: headers on the wire plus per-packet processing in the NIC and the kernel. On the wire, Ethernet adds 38 bytes beyond the IP packet: 8 preamble, 14 header, 4 FCS and a 12-byte inter-frame gap. With IPv4+TCP headers taking 40 bytes of the MTU:

| MTU | TCP payload per frame | Bytes on the wire | Payload efficiency | Frames/s to fill 100 Gbps |
|---|---|---|---|---|
| 1500 | 1460 | 1538 | 94.9% | ~8.1 million |
| 9000 | 8960 | 9038 | 99.1% | ~1.4 million |

The bandwidth gain is only about 4%. The bigger effect is in the last column. Handling 8 million packets per second in software is a lot of CPU work, which is one reason TCP at 100+ Gbps is hard (more on this below). NIC offloads like TSO/GRO hide some of the cost by letting the kernel deal with large chunks while the NIC splits and merges them.

The classic MTU failure is a **mismatch**. One host is set to 9000, a switch in the path is at 1500, and small packets (like ping or SSH handshakes) get through while large transfers hang or crawl. If "ping works but the transfer stalls", check MTU along the path:

```bash
ip link show eth0                  # look for "mtu 9000"
ping -M do -s 8972 other-host      # 8972 + 8 ICMP + 20 IP = 9000; -M do forbids fragmentation
```

If the large ping fails while a plain `ping other-host` succeeds, something in the path has a smaller MTU.

InfiniBand has its own MTU, usually 4096 bytes (IB supports 256 to 4096). RoCE deployments also commonly use a 4096-byte RDMA MTU on top of jumbo Ethernet frames.

## Ethernet, switches and routers

### MAC addresses and Ethernet

Every network interface card (NIC) has a **MAC address**, a 48-bit identifier written like `3c:fd:fe:a1:b2:c3`. It is burned in by the manufacturer, although it can be overridden. MAC addresses are flat: nothing in the address tells you where the device is. That is fine within one local network and useless at Internet scale, which is why IP exists.

An Ethernet II frame:

![Ethernet II frame format](images/ethernet-frame.png)

([source](https://commons.wikimedia.org/wiki/File:Ethernet_Type_II_Frame_format.svg))

Ethernet is the link layer that nearly every data center uses for general traffic. Speeds you will see: 1, 10, 25, 100, 200, 400 and 800 Gbps.

### Switches vs routers

A **switch** (L2) connects devices in the same local network, also called a broadcast domain or VLAN. It learns which MAC address sits behind which port by watching source addresses on incoming frames. It then forwards each frame only to the port where the destination lives. If it does not know the destination yet, it floods the frame to all ports.

A **router** (L3) connects different IP networks. It reads the destination IP, looks it up in a routing table and forwards the packet to the next hop. It also decrements the TTL so a packet caught in a loop eventually dies.

In modern data centers the line is blurry. "L3 switches" do both, at line rate, in hardware. What matters conceptually:

- Inside one subnet, hosts reach each other directly through switches using MAC addresses.
- Between subnets, traffic must go through a router (a gateway).

How does a host learn the MAC address for an IP on its own subnet? It uses **ARP** (Address Resolution Protocol). It broadcasts "who has 10.1.2.7?" and the owner answers with its MAC. `ip neigh` shows the resulting cache.

## IP addressing, subnets and routing

### IPv4 addresses and CIDR

An IPv4 address is 32 bits, written as four bytes: `10.1.2.7`. Addresses are grouped into **subnets** using **CIDR** notation: `10.1.2.0/24` means "the first 24 bits are the network, the remaining 8 bits pick a host". That gives 2^8 = 256 addresses, `10.1.2.0` to `10.1.2.255`.

| CIDR | Netmask | Addresses | Range |
|---|---|---|---|
| `10.1.2.0/24` | 255.255.255.0 | 256 | 10.1.2.0 – 10.1.2.255 |
| `10.1.0.0/22` | 255.255.252.0 | 1024 | 10.1.0.0 – 10.1.3.255 |
| `10.0.0.0/8` | 255.0.0.0 | 16,777,216 | 10.0.0.0 – 10.255.255.255 |

So `10.1.3.77` is inside `10.1.0.0/22` and `10.1.4.1` is not. The quickest way to check is with Python's standard library:

```python
import ipaddress
ipaddress.ip_address("10.1.3.77") in ipaddress.ip_network("10.1.0.0/22")  # True
ipaddress.ip_address("10.1.4.1")  in ipaddress.ip_network("10.1.0.0/22")  # False
```

In each subnet the first address is the network address and the last is the broadcast address, so neither can be assigned to a host. That leaves 254 usable addresses in a /24.

### Private ranges

Three ranges are reserved for private networks (RFC 1918). They are not routed on the public Internet, and cluster-internal networks almost always use them:

| Range | CIDR | Size |
|---|---|---|
| 10.0.0.0 – 10.255.255.255 | `10.0.0.0/8` | 16.7M |
| 172.16.0.0 – 172.31.255.255 | `172.16.0.0/12` | 1M |
| 192.168.0.0 – 192.168.255.255 | `192.168.0.0/16` | 65,536 |

A few other special ranges you will see:

- `127.0.0.0/8`: loopback. `127.0.0.1` is "this machine". Traffic never leaves the host.
- `169.254.0.0/16`: link-local. A host gives itself one of these when DHCP failed. On clouds, `169.254.169.254` is the instance metadata service.
- `172.17.0.0/16`: the default Docker bridge network. It is a private range like any other, but it often confuses NCCL because it looks like a real interface (see below).

IPv6 uses 128-bit addresses like `fe80::1`. Most HPC clusters still run IPv4 internally. If you see IPv6 in error messages, it is usually because a hostname resolved to an IPv6 address you were not expecting.

### Interfaces and the routing table

A machine has several **network interfaces**: loopback `lo`, one or more Ethernet NICs (`eth0`, `ens5`, `enp65s0f0` and similar), maybe InfiniBand IP interfaces (`ib0`), Docker bridges (`docker0`) and VPN tunnels. Each interface can have one or more IP addresses.

```bash
$ ip -br addr          # brief view: interface, state, addresses
lo               UNKNOWN        127.0.0.1/8 ::1/128
ens5             UP             10.1.2.7/24
ib0              UP             192.168.100.7/24
docker0          DOWN           172.17.0.1/16
```

A real GPU node often has many more: one management/frontend NIC plus 8 backend NICs, one per GPU. The output above is illustrative.

The **routing table** decides which interface and next hop each outgoing packet uses:

```bash
$ ip route
default via 10.1.2.1 dev ens5
10.1.2.0/24 dev ens5 proto kernel scope link src 10.1.2.7
192.168.100.0/24 dev ib0 proto kernel scope link src 192.168.100.7
```

Read it as follows. To reach `10.1.2.x`, send directly out of `ens5`. To reach `192.168.100.x`, send directly out of `ib0`. For anything else, send to the gateway `10.1.2.1` via `ens5`. When several routes match, the most specific one (longest prefix) wins.

To ask the kernel which route it would use for a given destination:

```bash
$ ip route get 192.168.100.9
192.168.100.9 dev ib0 src 192.168.100.7
```

This one command answers "will traffic to that node go over the fast network or the slow one?" It is worth remembering.

### DNS, /etc/hosts and hostname resolution

People and scripts use names (`node-017`). Sockets need IP addresses. **Name resolution** turns one into the other. On Linux the order is set by `/etc/nsswitch.conf` and is usually `files dns`: check `/etc/hosts` first, then ask a DNS server (configured in `/etc/resolv.conf`, often through `systemd-resolved`).

```bash
getent hosts node-017     # resolves the same way programs do (honors /etc/hosts + DNS)
hostname                  # this machine's name
hostname -I               # all of this machine's IPs (Linux)
```

Use `getent` rather than `nslookup` or `dig` when debugging cluster jobs. `nslookup` and `dig` query DNS directly and skip `/etc/hosts`, so they can disagree with what your program actually sees.

Hostname resolution failures that are typical on clusters:

1. **The hostname resolves to loopback.** Debian/Ubuntu images often ship with a line like `127.0.1.1 node-017` in `/etc/hosts`. If you then set `MASTER_ADDR=$(hostname)`, the master binds to or advertises `127.0.1.1`, and other nodes cannot connect to it. Check with `getent hosts $(hostname)`. If you see a 127.x address, use the real IP instead.
2. **The name resolves to the wrong interface.** The hostname maps to the frontend/management IP, while you wanted traffic on another network. For bootstrap this is usually fine. It becomes a problem when a firewall sits between the networks.
3. **Short vs fully qualified names.** `node-017` resolves but `node-017.cluster.local` does not, or the reverse, depending on the `search` line in `/etc/resolv.conf`.
4. **Inconsistent `/etc/hosts` across nodes.** The file was edited on one node but not on the others.
5. **DNS overload at startup.** Thousands of ranks resolve names at the same moment and the DNS server times out. It is rare, but it does happen at scale. Resolve once in the launcher and pass IPs around.

The SLURM idiom for choosing a master is:

```bash
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
```

This relies on that hostname resolving correctly from every node. Verify it with `getent hosts $MASTER_ADDR` on a worker.

## Transport: TCP and UDP

IP delivers packets between hosts, best effort. Packets can be lost, duplicated or reordered. The transport layer adds two things on top: **ports**, which identify which program on the host gets the data, and, in TCP's case, **reliability**.

### Ports and sockets

A **port** is a 16-bit number (0 to 65535). A connection is identified by a 5-tuple: (protocol, source IP, source port, destination IP, destination port). Servers listen on well-known ports, for example SSH on 22, HTTPS on 443, and torch distributed's default `MASTER_PORT` of 29500. Clients get a random **ephemeral port**. On Linux the default range is 32768 to 60999, visible in `/proc/sys/net/ipv4/ip_local_port_range`.

Ports below 1024 need root to bind. Only one process can listen on a given (IP, port, protocol) at a time. That rule is why "address already in use" happens when a previous training run is still alive, or a zombie is holding port 29500.

A **socket** is the operating system's handle for one endpoint of communication. Here is a complete TCP server and client in Python. This is the same pattern that sits under PyTorch's TCP store.

```python
# server.py -- run on the "master" node
import socket

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:   # IPv4, TCP
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)    # allow quick restart
    s.bind(("0.0.0.0", 29500))     # 0.0.0.0 = listen on all interfaces
    s.listen()
    print("listening on port 29500")
    conn, addr = s.accept()        # blocks until a client connects
    with conn:
        print("connection from", addr)
        while data := conn.recv(4096):   # b"" means the peer closed
            conn.sendall(data.upper())
```

```python
# client.py -- run on any node: python client.py <master-ip>
import socket, sys

host = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1"
with socket.create_connection((host, 29500), timeout=5) as s:
    s.sendall(b"hello from rank 1")
    print(s.recv(4096))            # b'HELLO FROM RANK 1'
```

This pair was tested on localhost. Across two nodes, a 5-second timeout on the client means one of these is true: a firewall dropped the packets, nothing is listening, or there is no route. A "Connection refused" means the host is reachable but nothing is listening on that port. That difference is a very useful diagnostic, and it comes back in the tools section.

Note the `0.0.0.0` in `bind`. If the server binds to `127.0.0.1` instead, only local clients can connect. A server bound to localhost is one of the most common reasons "it works on one node but not across nodes".

### The TCP handshake and reliability

TCP gives you a reliable, ordered byte stream on top of unreliable IP packets. Opening a connection takes a three-way handshake:

```
  client                                 server (listening)
    |  ---- SYN (seq=x) ---------------->  |
    |  <--- SYN-ACK (seq=y, ack=x+1) ----  |
    |  ---- ACK (ack=y+1) -------------->  |
    |        connection established        |
```

The three-way handshake:

![TCP three-way handshake](images/tcp-handshake.png)

([source](https://commons.wikimedia.org/wiki/File:Tcp-handshake.svg))

That costs one round trip before any data flows. After that, TCP:

- numbers every byte (sequence numbers),
- has the receiver acknowledge (ACK) what it received,
- retransmits anything not ACKed within a timeout, or after duplicate ACKs signal a gap,
- reorders out-of-order segments before handing the bytes to the application.

A firewall that silently drops packets causes the client's SYN to be retransmitted with exponential backoff. That is why a blocked port shows up as a long hang rather than an immediate error.

### Flow control and congestion control

TCP has two separate "slow down" mechanisms:

- **Flow control** protects the *receiver*. The receiver advertises a **window**: how many more bytes it has buffer space for. The sender may not have more than that much unacknowledged data in flight.
- **Congestion control** protects the *network*. The sender keeps a congestion window. It starts small (slow start), grows while ACKs come back, and shrinks sharply when it detects loss, which it takes as a sign of congestion. Linux defaults to CUBIC; BBR is a common alternative.

So a TCP sender can never have more than `min(receive window, congestion window)` bytes in flight. That leads directly to the most useful formula in networking.

### Latency, bandwidth, throughput

These three words get mixed up constantly:

- **Latency:** how long one small message takes to arrive, usually measured as round-trip time (RTT). It is set by distance (light in fiber covers about 200,000 km/s, so 100 m of fiber is about 0.5 µs one way and 4000 km is about 20 ms), switch hops, and software overhead.
- **Bandwidth:** the maximum rate the link can carry, such as 400 Gbps. It is a property of the pipe.
- **Throughput:** the rate you actually achieve. It is always at most the bandwidth, and often much less.

A useful first-order model for the time to send a message of `S` bytes:

```
time ≈ latency + S / bandwidth
```

Small messages are dominated by latency, large ones by bandwidth. Sending 1 GiB over a 400 Gbps link takes at least 2^30 × 8 / 400e9 ≈ 21.5 ms, and the latency term is negligible. Sending 8 bytes takes as long as the latency, whatever the bandwidth. This is why the benchmark chapters plot bandwidth as a function of message size: small messages never reach the peak.

### Bandwidth-delay product

To keep a pipe full you need enough data in flight to cover a full round trip. The **bandwidth-delay product** (BDP) is:

```
BDP (bytes) = bandwidth (bits/s) × RTT (s) / 8
```

Worked numbers:

| Link | Bandwidth | RTT | BDP |
|---|---|---|---|
| Home connection | 1 Gbps | 20 ms | 2.5 MB |
| Data center 10GbE | 10 Gbps | 0.2 ms | 250 KB |
| InfiniBand NDR, same rack | 400 Gbps | 5 µs | 250 KB |
| Cross-region cloud link | 100 Gbps | 70 ms | 875 MB |

The implication: if a TCP connection's window is capped at 64 KiB, throughput is capped at window/RTT. That is 65536 × 8 / 0.2 ms ≈ 2.6 Gbps in the data center and only about 7.5 Mbps across a 70 ms link, no matter how fat the pipe is. Linux autotunes socket buffers. The typical default maxima are around 6 MB receive and 4 MB send (`net.ipv4.tcp_rmem` / `tcp_wmem`; check your distro). That is plenty for the data center but far short of the 875 MB a cross-region 100 Gbps link would need. This is why copying a checkpoint across regions over a single TCP stream is slow, and why tools open many parallel streams.

The InfiniBand row is worth a second look. The BDP is small because the latency is tiny. On that fabric, software overhead per message, not the window, is the bottleneck.

### UDP

**UDP** is the minimal transport: ports plus a checksum, and nothing else. There is no handshake, no ordering, no retransmission and no congestion control. Each `send` is one datagram that may or may not arrive. It is used where the application wants control: DNS, video, QUIC (HTTP/3), and, importantly for us, **RoCEv2, which encapsulates InfiniBand transport packets inside UDP** (destination port 4791). UDP is used there purely as a cheap, routable wrapper. Reliability is provided by the RDMA transport inside it.

### Why TCP is slow for HPC

TCP is excellent for the Internet. For moving gradients between GPUs it has three structural costs:

1. **Copies.** Sending GPU memory over a TCP socket goes like this: GPU memory → host memory (cudaMemcpy) → kernel socket buffer (copy during `send`) → NIC (DMA). The receive side repeats it in reverse. Each copy uses memory bandwidth and adds latency.
2. **CPU and kernel overhead.** Every send and receive is a system call. The kernel runs the TCP state machine, computes checksums (unless offloaded) and handles interrupts. At 100 Gbps and 1500 MTU that is about 8 million packets per second. A single TCP stream typically cannot saturate a 100+ Gbps NIC on one core, so you need many streams and many cores. Those cores are the same ones running your data loader.
3. **Latency floor.** A round trip through the kernel stack on a LAN is typically tens of microseconds. RDMA hardware achieves low single-digit microseconds. Exact numbers depend heavily on hardware and tuning.

NCCL can run over plain TCP sockets (its "Socket" transport). When the RDMA path is misconfigured, NCCL may silently fall back to it, and all-reduce bandwidth drops by an order of magnitude or more. Spotting `NET/Socket` instead of `NET/IB` in `NCCL_DEBUG=INFO` output is one of the most valuable debugging skills covered in [network debug](../network/debug/README.md).

## Units: reading spec sheets without getting fooled

Networking people and ML people use different units, and vendors take advantage of the confusion.

**Bits vs bytes.** Network links are quoted in **bits per second** with decimal prefixes: 400 Gbps = 400 × 10^9 bits/s. Memory and file sizes are in bytes. Divide by 8:

```
400 Gbps NIC          = 50 GB/s
8 x 400 Gbps per node = 3200 Gbps = 400 GB/s
```

Watch the capitalization. `Gb` is gigabit and `GB` is gigabyte. Some benchmark tools also report GiB/s (2^30 bytes/s), which is about 7% less than GB/s for the same number.

**Line rate vs payload.** The quoted rate is the raw signaling rate after line encoding. Packet headers, as in the MTU table above, plus protocol overhead mean that application payload always comes out a few percent lower. A well-tuned 400 Gbps InfiniBand link typically delivers somewhere in the high 300s Gbps in `ib_write_bw`. Do not expect 400.

**Unidirectional vs bidirectional.** A full-duplex link can send at full rate in both directions at once. Many vendors quote the *sum of both directions*. NVIDIA's "900 GB/s NVLink" for H100 is bidirectional: it is 450 GB/s in each direction. A ring all-reduce sends in one direction on each link, so the per-direction number is the one that matters. The [network chapter](../network/README.md#unidirectional-vs-bidirectional-duplex) covers this in detail. **Always ask "per direction or both?"**

**Aggregate vs per-link.** "3.2 Tbps per node" may mean 8 NICs × 400 Gbps. One process talking to one peer through one NIC gets only one NIC's worth.

**Theoretical vs achieved.** Peak numbers assume large messages, perfect topology and no congestion. Real collectives on real jobs achieve less. That is why the book stresses measuring with [benchmarks](../network/benchmarks/README.md) rather than trusting the brochure.

## Practical tools

Most of these are installed by default or one package away. All examples assume Linux. On your own machines, install `iperf3`, `netcat-openbsd` (or `ncat`), `tcpdump` and `traceroute` through your package manager.

### ping

```bash
ping -c 4 node-017
```

`ping` sends ICMP echo requests and reports round-trip time. It answers "is the host reachable at the IP layer, and what is the latency?" Caveat: many cloud environments and firewalls block ICMP, so a failed ping does not prove the host is unreachable. Use `nc` (below) to test a TCP port directly.

### traceroute / tracepath

```bash
traceroute -n node-017     # -n: don't resolve names, faster
tracepath -n node-017      # no root needed, also discovers path MTU
```

These show each router hop between you and the target. Inside one subnet there is just one hop. Use them to answer "is my traffic leaving through the gateway I expect?"

### ss (and netstat)

`ss` is the modern replacement for `netstat`:

```bash
ss -tlnp                   # TCP, listening, numeric, show process (-p needs root for others' processes)
ss -tnp state established  # current TCP connections
ss -tlnp | grep 29500      # is anything (e.g. a stale rank-0) listening on MASTER_PORT?
```

If `ss -tlnp` shows your server listening on `127.0.0.1:29500` instead of `0.0.0.0:29500` or the node's IP, other nodes cannot reach it.

### nc (netcat)

The Swiss-army knife for testing TCP/UDP connectivity:

```bash
# on node A: listen on a port
nc -l 29500                # OpenBSD netcat; some variants need: nc -l -p 29500

# on node B: test whether the port is reachable
nc -zv nodeA 29500         # -z: just scan, -v: verbose
```

Read the three outcomes like this:

- `succeeded` / `open`: the path works.
- `Connection refused`: the host is reachable, but nothing is listening, or a firewall actively rejected the connection.
- Hang, then timeout: the packets are being dropped. Usually a firewall, a security group, or no route.

This is the fastest way to tell whether `MASTER_PORT` is reachable before you blame PyTorch.

### iperf3

Measures TCP/UDP throughput between two hosts:

```bash
iperf3 -s                       # on server
iperf3 -c server -t 10          # on client: single stream, 10 seconds
iperf3 -c server -t 10 -P 8     # 8 parallel streams
iperf3 -c server -R             # reverse direction (server sends)
```

If one stream gets 20 Gbps on a 100 Gbps NIC and 8 streams get 90, the limit is per-stream CPU or window, not the wire. `iperf3` only tests the TCP/IP path. To test RDMA, use the `perftest` suite (`ib_write_bw`, `ib_read_lat`), and to test the whole GPU communication stack use `nccl-tests` or the book's [torch-dist-bench](../network/benchmarks/README.md#torch-dist-bench).

### curl

```bash
curl -v http://localhost:6006/              # is TensorBoard up? -v shows the connection + headers
curl -sS -o /dev/null -w '%{http_code} %{time_total}\n' https://huggingface.co
curl -s http://169.254.169.254/latest/meta-data/   # AWS metadata (IMDSv1; IMDSv2 requires a token)
```

`curl -v` shows DNS resolution, the TCP connection, the TLS handshake and HTTP status, layer by layer. That makes it a good example of the layering in action.

### tcpdump basics

`tcpdump` captures packets on an interface. It needs root.

```bash
sudo tcpdump -i ens5 -nn port 29500           # watch traffic on MASTER_PORT
sudo tcpdump -i ens5 -nn 'tcp[tcpflags] & tcp-syn != 0'   # only SYN packets (connection attempts)
sudo tcpdump -i ens5 -nn -w capture.pcap      # save to file, open later in Wireshark
```

Debugging a hanging rendezvous: run tcpdump on the master and start a worker. If SYNs arrive but no SYN-ACK goes back, the master's firewall or listener is the problem. If no SYNs arrive at all, the problem is on the worker side or in the network between. tcpdump does **not** see RDMA traffic. RDMA bypasses the kernel, which is exactly the point.

### ssh: keys, config, tunnels

You will use SSH constantly, and multi-node launchers (pdsh, the DeepSpeed launcher, MPI) often use it to start processes on other nodes.

**Keys.** Use key pairs instead of passwords:

```bash
ssh-keygen -t ed25519                  # creates ~/.ssh/id_ed25519 (private) and .pub (public)
ssh-copy-id user@login.cluster.org     # appends your public key to the remote ~/.ssh/authorized_keys
```

Launchers that ssh between compute nodes need passwordless SSH from node to node. On clusters with a shared home directory, that means your own public key must be in your own `~/.ssh/authorized_keys`.

**Config.** Put repeated options in `~/.ssh/config`:

```
Host login
    HostName login.cluster.org
    User ibrahim
    IdentityFile ~/.ssh/id_ed25519

Host gpu-*
    ProxyJump login              # reach compute nodes through the login node
    User ibrahim
```

Now `ssh gpu-017` hops through the login node automatically.

**Port forwarding (tunnels).** Jupyter (8888) and TensorBoard (6006) on a compute node listen on that node's localhost. Compute nodes are normally not reachable from your laptop. A **local port forward** fixes this:

```bash
# laptop's localhost:8888  -->  (via login)  -->  gpu-017's localhost:8888
ssh -N -L 8888:localhost:8888 gpu-017
# then open http://localhost:8888 on the laptop
```

`-L local_port:target_host:target_port` means "listen on my local port, and forward each connection through the SSH session to `target_host:target_port` *as seen from the remote end*". `-N` means "do not run a remote command, just forward". Two variations are worth knowing. `-L 6006:gpu-017:6006 login` forwards through the login node to the compute node, which works only if the service on gpu-017 listens on a non-localhost interface. `-R` (remote forward) works in the opposite direction.

This is far safer than starting Jupyter with `--ip=0.0.0.0` on a shared cluster, which exposes it to everyone on the network.

### Firewalls and why MASTER_PORT gets blocked

A **firewall** filters packets by rules on IPs, ports and protocols. On Linux the host firewall is `nftables`, often managed through `ufw` (Ubuntu) or `firewalld` (RHEL). In clouds there is another layer outside the host: **security groups** or **VPC firewall rules**.

Common ways distributed training trips over firewalls:

- Only SSH (22) is allowed between nodes, so `MASTER_PORT=29500` hangs at rendezvous.
- The master port is open, but **NCCL and Gloo open more connections on random ephemeral ports** after bootstrap. If only 29500 is open, rendezvous succeeds and then NCCL init hangs or fails. The usual fix on a trusted cluster network is to allow all traffic between cluster nodes (for example, a security group that allows all traffic from itself).
- On AWS EFA specifically, the security group must allow all inbound and outbound traffic to and from itself. EFA's own traffic is not TCP, but it still goes through the security group.

To check the host firewall:

```bash
sudo nft list ruleset          # raw rules
sudo ufw status                # Ubuntu
sudo firewall-cmd --list-all   # RHEL/Rocky
```

## Data center networking

Now zoom out from one link to the whole building.

### Top-of-rack, leaf-spine, fat-tree

Servers sit in racks. Each rack traditionally has a **top-of-rack (ToR)** switch that all servers in the rack connect to. In modern terms the ToR is a **leaf**. Every leaf connects to every **spine** switch:

```
             +--------+    +--------+    +--------+    +--------+
  spine:     | spine1 |    | spine2 |    | spine3 |    | spine4 |
             +--------+    +--------+    +--------+    +--------+
                  \\\\        \\\\  ////       ////
                   full mesh: every leaf has one uplink
                   to every spine (4 uplinks per leaf here)
                  ////        ////  \\\\       \\\\
             +--------+    +--------+    +--------+    +--------+
  leaf/ToR:  | leaf1  |    | leaf2  |    | leaf3  |    | leaf4  |
             +--------+    +--------+    +--------+    +--------+
              | | | |       | | | |       | | | |       | | | |
             servers       servers       servers       servers
```

A fat tree:

![fat tree network](../network/images/fat-tree-network.png)

([source](https://commons.wikimedia.org/wiki/File:Fat_tree_network.svg))

Properties that matter:

- Any two servers on *different* leaves are exactly 3 switch hops apart: leaf → spine → leaf. Latency is predictable.
- There are as many equal-length paths between two leaves as there are spines. That is the raw material for ECMP (below).
- To grow, you add leaves. To get more bandwidth between leaves, you add spines.

The size limit comes from switch port counts. With 64-port switches, a two-tier non-blocking leaf-spine gives each leaf 32 ports down and 32 up, and each spine can connect to 64 leaves. That caps it at 64 × 32 = 2048 servers. Beyond that you add a third tier.

A **fat-tree** (a folded Clos network) is the general multi-tier version of this idea. Links get "fatter" (more aggregate bandwidth) toward the root, so the bandwidth between any two halves of the network stays constant. A classic 3-tier fat-tree built from k-port switches connects k^3/4 hosts at full bandwidth. For k=64 that is 65,536. InfiniBand AI clusters are commonly built as fat-trees.

AI clusters add a twist: **rail-optimized** topologies. In an 8-GPU node, NIC 0 of every node connects to the "rail 0" leaf switches, NIC 1 to rail 1, and so on. GPU *i* on one node reaches GPU *i* on another node in a single leaf hop, and NCCL arranges its traffic to exploit this.

### Oversubscription

A leaf's **oversubscription ratio** is downlink bandwidth to servers divided by uplink bandwidth to spines. A 64-port leaf with 48 ports down to servers and 16 ports up to spines (all at the same speed) is **3:1 oversubscribed**. If every server sends off-rack at full rate at the same time, each gets only one third.

For web traffic, where most servers are idle most of the time, 3:1 is cheap and fine. For training, where *every* GPU communicates at the same moment during an all-reduce, oversubscription directly cuts collective bandwidth. AI backend fabrics are therefore usually built **non-blocking (1:1)**, or close to it, at least within a job's scale unit. Ask the provider. It is one of the most important questions about a cluster you are renting.

### ECMP

**ECMP** (Equal-Cost Multi-Path) is how a leaf spreads traffic across its many uplinks. Hashing each packet's 5-tuple (src/dst IP, src/dst port, protocol) picks one of the equal-cost paths. All packets of one flow take the same path, which keeps them in order, and different flows spread out.

The catch for ML: ECMP balances *flows*, not bytes. A training job has a few very large, long-lived flows (sometimes called "elephant flows"). If two elephants happen to hash onto the same uplink, that link is congested while others sit idle. With few flows, bad luck is likely. Mitigations include more connections per peer (for example `NCCL_IB_QPS_PER_CONNECTION`, covered in [benchmarks](../network/benchmarks/README.md#nccl_ib_qps_per_connection)), InfiniBand **adaptive routing**, which picks paths per packet based on congestion, and newer Ethernet designs that spray packets across all paths.

### East-west vs north-south

- **North-south** traffic enters or leaves the data center: users hitting your inference API, downloading datasets from the Internet, pushing to the Hub.
- **East-west** traffic stays between servers inside the data center: gradient all-reduce, reading from a parallel filesystem, checkpoint writes.

Training is overwhelmingly east-west. Clusters often physically separate these into a **frontend network** (Ethernet: SSH, storage, the Internet) and a **backend/compute network** (InfiniBand or RoCE, GPU-to-GPU only), plus an **out-of-band management network** for BMCs. The [network chapter](../network/README.md#cluster-networks) describes these. When debugging, always know which of these networks a given connection is supposed to use.

## HPC networking: RDMA, InfiniBand, RoCE, EFA

### What RDMA removes

**RDMA** (Remote Direct Memory Access) lets the NIC on one machine read or write memory on another machine directly. The remote CPU is not involved, and on the data path the operating system kernel is not involved on either side.

Compare the TCP send path with the RDMA one:

```
TCP:                                         RDMA:
 app buffer                                   app buffer (pre-registered)
   | copy (send syscall, CPU)                   |
 kernel socket buffer                           | NIC DMAs directly from app memory
   | TCP/IP processing (CPU)                    | (no syscall, no copy, no kernel)
 NIC --- wire --- NIC                         NIC --- wire --- NIC
   | interrupt, TCP processing (CPU)            | NIC DMAs directly into app memory
 kernel socket buffer                           |
   | copy (recv syscall, CPU)                 app buffer (pre-registered)
 app buffer
```

RDMA removes four things:

1. **Kernel involvement on the data path** (kernel bypass). The application posts work requests to queues that the NIC reads directly.
2. **Intermediate copies** (zero-copy). The NIC DMAs to and from the application's own buffers.
3. **Remote CPU involvement.** In a one-sided RDMA WRITE, the remote CPU does not even learn that its memory was written, unless it checks.
4. **Software transport processing.** Reliability, ordering and segmentation are implemented in NIC hardware.

The price is setup complexity. Memory must be **registered** (pinned and made known to the NIC) before use. Connections are set up out of band, which is where TCP comes back in during bootstrap. The programming model is much lower level than sockets.

### InfiniBand

**InfiniBand (IB)** is a networking technology designed from scratch for HPC, with RDMA as its native mode. NVIDIA (through its acquisition of Mellanox) is effectively the only vendor. The key pieces:

- **HCA** (Host Channel Adapter): the IB NIC, for example a ConnectX card. It shows up as a device such as `mlx5_0`.
- **Subnet Manager (SM):** software, either OpenSM or NVIDIA UFM, running on a switch or a host. It discovers the fabric topology, assigns addresses, and computes forwarding tables for all switches. Without a running SM, IB ports stay in the `INIT` state and do not pass traffic. "Port state INIT" means "go check the SM".
- **LID** (Local Identifier): a 16-bit address the SM assigns to each port. It plays the role of a MAC/IP address inside an IB subnet. Ports also have permanent 64-bit **GUIDs**, similar to MAC addresses.
- **Verbs:** the IB/RDMA programming interface (`libibverbs`). The core objects are the **Queue Pair (QP)**, a send queue plus a receive queue representing one connection, the **Completion Queue (CQ)**, where the NIC reports finished work, and **Memory Regions (MR)**, which are registered buffers. NCCL, MPI and UCX all speak verbs. You will rarely write verbs code yourself, but `NCCL_IB_*` environment variables and error messages use these terms.
- **Credit-based flow control:** IB is **lossless** at the link layer. A sender transmits only when the receiver has advertised buffer credits, so switches do not drop packets because of congestion. This is a big reason IB performance is so predictable.

IB link generations are named by per-lane speed: EDR, HDR, NDR, XDR. A typical port today is 4 lanes, for example NDR 4x = 400 Gbps. The [network chapter](../network/README.md#infiniband) has the table.

Useful commands (from the `infiniband-diags` and `rdma-core` packages):

```bash
ibstat                 # HCA ports: State: Active, Rate: 400, LID
ibv_devinfo            # verbs-level view of devices
ibv_devices            # list RDMA devices
rdma link show         # iproute2 view of RDMA links (works for IB and RoCE)
```

IB can also carry IP traffic (**IPoIB**, the `ib0` interface). IPoIB is convenient for bootstrap and SSH, but it is not the fast path. NCCL uses verbs directly.

### RoCE and lossless Ethernet

**RoCE** (RDMA over Converged Ethernet) runs the InfiniBand transport over Ethernet. RoCEv1 is plain L2 Ethernet and not routable. **RoCEv2** wraps it in UDP/IP so it can be routed, and RoCEv2 is what everyone means today. You get RDMA semantics using Ethernet switches and Ethernet operational knowledge.

The problem: Ethernet drops packets when buffers fill, and the RDMA transport in NICs historically handles loss poorly. Its recovery is crude (go-back-N), and throughput collapses under loss. RoCE deployments therefore try to make Ethernet behave **losslessly**, using two mechanisms:

- **PFC** (Priority Flow Control, IEEE 802.1Qbb). When a switch buffer for a traffic class nears full, it sends a "pause" frame upstream, telling the sender to stop that class briefly. It prevents drops, but pauses can propagate backwards and cause "head-of-line blocking" and, in pathological cases, deadlock. It is a blunt tool.
- **ECN** (Explicit Congestion Notification). Instead of dropping, a congested switch *marks* packets. The receiver reflects the mark back to the sender, which reduces its rate. On RoCE NICs this loop is usually **DCQCN**. ECN acts earlier and more gently than PFC. PFC stays as a backstop.

Getting PFC/ECN configured consistently on every NIC and switch is the main operational burden of RoCE. When it is wrong, the symptoms are mysterious tail-latency spikes and bandwidth collapse under load. Newer designs, such as the Ultra Ethernet Consortium specifications and vendor-specific NICs, aim to tolerate loss and reduce the dependence on PFC. This area is evolving, so treat any specific claim about it as dated.

### GPUDirect RDMA

Plain RDMA still lands data in *host* memory. For GPU data that adds a copy: GPU memory → host memory → NIC. **GPUDirect RDMA** lets the NIC DMA directly to and from **GPU memory** over PCIe, so the bytes never touch host RAM:

```
without GPUDirect RDMA:    GPU mem --PCIe--> host RAM --PCIe--> NIC --> wire
with GPUDirect RDMA:       GPU mem --PCIe (through a PCIe switch)--> NIC --> wire
```

Requirements: a supported NIC and GPU, and a kernel module or mechanism that lets the NIC driver access GPU memory. That is the `nvidia-peermem` module, or the newer dma-buf path on recent drivers and kernels. Placement also matters: it works best when the GPU and NIC sit under the same PCIe switch. In NCCL logs, GPUDirect RDMA shows up as `GDRDMA` on the transport line, and `NCCL_NET_GDR_LEVEL` controls when NCCL uses it. If the module is missing, NCCL quietly stages through host memory and runs slower.

### EFA (AWS)

AWS does not offer InfiniBand. Its HPC network is **EFA** (Elastic Fabric Adapter). EFA provides OS-bypass networking using AWS's own **SRD** (Scalable Reliable Datagram) protocol. SRD sprays packets across many paths and handles reordering and retransmission in the NIC. That is a different design choice from IB and RoCE: tolerate loss and spread load, rather than make the network lossless.

Software-wise, EFA is accessed through **libfabric**, and NCCL reaches it through the **aws-ofi-nccl** plugin. Practical consequences:

- NCCL logs show `NET/OFI` (the plugin) instead of `NET/IB`.
- You need the EFA driver, libfabric and the plugin installed in your container, at compatible versions. A missing plugin means NCCL falls back to TCP sockets, and that fallback is a very common cause of "training on AWS is mysteriously slow".
- `fi_info -p efa` lists EFA devices. If it shows nothing, EFA is not usable from this environment.
- As noted above, the security group must allow all traffic within itself.

Other clouds have their own equivalents. Google Cloud, for example, has GPUDirect-TCPX/TCPXO and RDMA-capable offerings on newer instance types. These details change frequently, so check the provider's current docs.

## Inside the node: a network inside the box

Everything so far was *between* machines. A modern 8-GPU server also has a high-performance network *inside* it, and for many workloads that is where most of the communication happens.

```
          +------------------------- one 8-GPU node --------------------------+
          |                                                                   |
          |   GPU0  GPU1  GPU2  GPU3  GPU4  GPU5  GPU6  GPU7                  |
          |    ||    ||    ||    ||    ||    ||    ||    ||    NVLink         |
          |  +-----------------------------------------------+               |
          |  |         NVSwitch chips (all-to-all fabric)     |               |
          |  +-----------------------------------------------+               |
          |    |     |     |     |     |     |     |     |      PCIe         |
          |  [PCIe switch]..[PCIe switch]..  each with GPU(s) + NIC(s)        |
          |       |               |                                           |
          |      CPU0 ========== CPU1    (CPU-CPU link: UPI / Infinity Fabric)|
          |       |               |                                           |
          |     NIC0..NIC3      NIC4..NIC7  ---> to the backend fabric        |
          +-------------------------------------------------------------------+
```

(A schematic. Real layouts vary by vendor and model; see `nvidia-smi topo -m` for yours.)

### PCIe

**PCIe** is the general-purpose bus connecting the CPU to GPUs, NICs and NVMe drives. It is a point-to-point, packet-based link organized as a tree, with PCIe switches in the middle. Approximate per-direction bandwidth for an x16 slot is about 32 GB/s for Gen4, about 64 GB/s for Gen5 and about 128 GB/s for Gen6, before protocol overhead.

Compare that to a 400 Gbps NIC: 50 GB/s needs a Gen5 x16 slot to avoid being bottlenecked by PCIe. This is why NIC-GPU-PCIe-switch placement matters. If a GPU's traffic to its NIC has to cross the CPU, or worse, cross between two CPU sockets, bandwidth drops and latency rises. `nvidia-smi topo -m` prints the relationships: `PIX` (same PCIe switch) is best, `SYS` (crosses the inter-socket link) is worst.

### NVLink and NVSwitch

**NVLink** is NVIDIA's dedicated GPU-to-GPU link, far faster than PCIe. H100 has 18 NVLink 4 links totaling 900 GB/s bidirectional (450 GB/s per direction). B200 doubles that. **NVSwitch** chips connect all GPUs in a node, so any GPU can talk to any other at full NVLink bandwidth. It is literally a switch, like the leaf switches above, but inside the box, carrying load/store and copy traffic between GPU memories.

Compare the numbers per GPU and per direction: about 450 GB/s over NVLink inside the node versus about 50 GB/s through a 400 Gbps NIC between nodes. That is roughly a 9x gap, and it is why parallelism strategies put the chattiest communication (tensor parallelism) inside the NVLink domain and the less frequent communication (data parallelism, pipeline stages) across nodes. Rack-scale systems such as NVL72 extend the NVLink domain to 72 GPUs across a rack, which shifts that boundary. AMD has the equivalent Infinity Fabric/xGMI. See the [intra-node section](../network/README.md#intra-node-networking).

NCCL picks the intra-node transport automatically:

- **P2P**: direct GPU-to-GPU over NVLink, or over PCIe when the topology allows.
- **SHM**: through host shared memory, when P2P is not possible.
- **NET**: through the NIC, between nodes.

## How PyTorch distributed bootstraps

Now we can follow what happens when you run a multi-node job, for example:

```bash
# on each of 2 nodes
torchrun --nnodes=2 --nproc-per-node=8 --node-rank=$NODE_RANK \
         --master-addr=10.1.2.7 --master-port=29500 train.py
```

and `train.py` calls `torch.distributed.init_process_group("nccl")`.

### Step 1: rendezvous over TCP (the store)

The processes need to find each other first. Nobody yet knows the others' addresses except for one agreed point: **`MASTER_ADDR:MASTER_PORT`**. There, a **TCPStore** runs. It is a simple key-value server over plain TCP, conceptually just like the socket example above.

```
      node 0 (10.1.2.7)                         node 1
  +-------------------------+          +-------------------------+
  | rank 0                  |          | ranks 8..15             |
  |  TCPStore server :29500 | <--TCP-- |  TCPStore clients       |
  | ranks 1..7 (clients)    |          |                         |
  +-------------------------+          +-------------------------+
       set/get keys: "who is out there?", "here is the NCCL unique id"
```

Every rank connects to the store. The world size is checked, and processes block until all ranks have arrived. That blocking is the hang you see when one node never shows up. With `torchrun`, the elastic agent manages this rendezvous; the details depend on `--rdzv-backend`. With a plain `init_method="env://"`, the `MASTER_ADDR`, `MASTER_PORT`, `RANK` and `WORLD_SIZE` environment variables drive it.

The store is low-bandwidth control traffic. Its only requirements are that every rank can open a TCP connection to `MASTER_ADDR:MASTER_PORT`, and that the master listens on an address the others can reach.

### Step 2: NCCL builds its own connections

When the first collective runs on a process group (or when it is eagerly initialized), NCCL initializes its *communicator*:

1. Rank 0 creates an `ncclUniqueId`, which contains the address of NCCL's own bootstrap socket. It publishes the id through the TCPStore.
2. Every rank fetches the id and connects to NCCL's **bootstrap network**, a TCP socket ring separate from the TCPStore and on different, ephemeral ports. Over it, ranks exchange their connection info and topology: which GPUs, which NICs, NVLink or PCIe.
3. NCCL computes rings and trees over the topology and establishes the **data connections**: P2P/SHM inside the node, and IB verbs QPs, EFA endpoints, or, as a fallback, TCP sockets between nodes.

Only after that does data flow, over the fast transports. So there are three layers of connectivity that can each fail independently:

| Phase | Transport | Ports | Fails as |
|---|---|---|---|
| TCPStore rendezvous | TCP | `MASTER_PORT` | Hang or timeout in `init_process_group` |
| NCCL bootstrap | TCP | ephemeral | Hang or error at first collective, "socket" errors |
| NCCL data | IB/RoCE/EFA/NVLink (or TCP fallback) | n/a for RDMA | Errors at first collective, or just slow |

### Which interface does NCCL pick?

For its bootstrap sockets, and for the Socket transport if it falls back to it, NCCL must pick an IP interface. By default it prefers interfaces whose name starts with `ib`, then any other interface except `lo` (loopback) and `docker*`. That heuristic goes wrong when a node has several interfaces: a VPN tunnel, a virtual bridge, a management NIC that cannot reach the other nodes. Then you override it:

```bash
export NCCL_SOCKET_IFNAME=ens5          # use exactly this interface (prefix match by default)
export NCCL_SOCKET_IFNAME=^docker,lo    # anything except these prefixes
export GLOO_SOCKET_IFNAME=ens5          # the same for the Gloo backend (CPU collectives)
export NCCL_IB_HCA=mlx5_0,mlx5_1        # which RDMA devices to use for the data path
```

`NCCL_SOCKET_IFNAME` selects an *IP interface* (from `ip addr`). `NCCL_IB_HCA` selects *RDMA devices* (from `ibv_devices`). They are different namespaces, and mixing them up is a common mistake. See [network debug](../network/debug/README.md#nccl_socket_ifname) for details.

### Common failures, mapped to layers

| Symptom | Likely layer | First check |
|---|---|---|
| Hangs in `init_process_group`, then times out | TCP to `MASTER_PORT` | `nc -zv $MASTER_ADDR $MASTER_PORT` from a worker; firewall; `ss -tlnp` on master |
| `Address already in use` | Port | Stale process on master: `ss -tlnp \| grep 29500`; choose another port |
| Workers connect to `127.0.1.1` / wrong IP | Name resolution | `getent hosts $MASTER_ADDR` on each node |
| Rendezvous OK, first all-reduce hangs | NCCL bootstrap / interface | `NCCL_DEBUG=INFO`; which interface it chose; set `NCCL_SOCKET_IFNAME`; allow all intra-cluster ports |
| Works, but all-reduce bandwidth is tiny | Transport fallback | `NCCL_DEBUG=INFO`: look for `NET/Socket` vs `NET/IB` / `NET/OFI`; `GDRDMA` present? |
| Works on 1 node, fails on 2 | Inter-node path | Intra-node uses NVLink and never touches the network; test with `nc`, `ib_write_bw` |
| Works in bare metal, fails in Docker | Interfaces/devices in container | `--network=host`, pass RDMA devices into the container, `--ipc=host`/shm size |
| Random NCCL timeouts mid-run | Fabric | Link errors, congestion; see [network debug](../network/debug/README.md) |

A good habit for the first run on any new cluster is to run a minimal all-reduce test before the real training job. The [torch-dist-bench](../network/benchmarks/README.md#torch-dist-bench) and the [network debug](../network/debug/README.md) chapter provide exactly that.

## Collective communication: an intuitive introduction

Point-to-point communication is "rank A sends a tensor to rank B". Distributed training mostly uses **collectives**, where every rank in a group takes part and the result is defined over all of them. The [communication patterns](../network/comms.md) chapter has diagrams for each. Here is what each one *computes*, with tiny arrays. Let there be 3 ranks, each holding a vector:

```
rank 0: [ 1,   2,   3 ]
rank 1: [10,  20,  30 ]
rank 2: [100, 200, 300]
```

**Broadcast** (root = rank 0): every rank ends with the root's data.

```
all ranks: [1, 2, 3]
```

Used for distributing initial weights.

**Reduce** (sum, to root 0): the root gets the elementwise sum. The others get nothing new.

```
rank 0: [111, 222, 333]
```

**All-reduce** (sum): *every* rank gets the elementwise sum.

```
all ranks: [111, 222, 333]
```

This is data-parallel gradient synchronization. Divide by 3 and you have averaged gradients.

**All-gather:** every rank gets everyone's pieces, concatenated in rank order.

```
all ranks: [1, 2, 3, 10, 20, 30, 100, 200, 300]
```

ZeRO/FSDP uses it to reassemble sharded parameters before a layer runs.

**Reduce-scatter** (sum): reduce, then split the result so each rank keeps one chunk.

```
rank 0: [111]
rank 1: [222]
rank 2: [333]
```

FSDP uses it for gradients: each rank keeps only the summed gradient shard it owns.

The key identity is **all-reduce = reduce-scatter followed by all-gather**. Reduce-scatter leaves each rank with one reduced chunk, and all-gather spreads those chunks to everyone. This is exactly how the ring algorithm works, and it is why FSDP can do "all-reduce in two halves" with compute in between.

**All-to-all:** rank *i* sends its *j*-th chunk to rank *j*, like a matrix transpose across ranks. It is used for expert parallelism in mixture-of-experts models.

```
rank 0: [1, 10, 100]
rank 1: [2, 20, 200]
rank 2: [3, 30, 300]
```

### Ring all-reduce, step by step

How do you compute an all-reduce without sending everything to one rank, which would make that rank's link the bottleneck? Arrange the ranks in a ring, `0 → 1 → 2 → 0`, where each rank only sends to its right neighbor. Split each vector into N=3 chunks (here each chunk is one number). Call the chunks c0, c1, c2.

The same idea as a picture, one chunk travelling around the ring:

![ring all-reduce chunk movement](../network/images/all-reduce-ring-chunk1.png)

([source](https://images.nvidia.com/events/sc15/pdfs/NCCL-Woolley.pdf))

**Phase 1: reduce-scatter (N−1 = 2 steps).** In step *k*, rank *r* sends chunk `(r − k) mod N` to rank `r+1`, which adds it to its own copy.

```
start:            rank0 [  1,   2,   3]   rank1 [ 10,  20,  30]   rank2 [100, 200, 300]

step 1 sends:     r0 -> r1: c0=1      r1 -> r2: c1=20      r2 -> r0: c2=300
after step 1:     rank0 [  1,   2, 303]   rank1 [ 11,  20,  30]   rank2 [100, 220, 300]

step 2 sends:     r0 -> r1: c2=303    r1 -> r2: c0=11      r2 -> r0: c1=220
after step 2:     rank0 [  1, 222, 303]   rank1 [ 11,  20, 333]   rank2 [111, 220, 300]
                               ^^^                         ^^^         ^^^
```

Each rank now owns one *fully reduced* chunk: rank 0 has c1=222, rank 1 has c2=333 and rank 2 has c0=111. That is a reduce-scatter.

**Phase 2: all-gather (N−1 = 2 steps).** Each rank passes its completed chunk around the ring. Receivers *overwrite* their copy instead of adding.

```
step 1 sends:     r0 -> r1: c1=222    r1 -> r2: c2=333     r2 -> r0: c0=111
after step 1:     rank0 [111, 222, 303]   rank1 [ 11, 222, 333]   rank2 [111, 220, 333]

step 2 sends:     r0 -> r1: c0=111    r1 -> r2: c1=222     r2 -> r0: c2=333
after step 2:     rank0 [111, 222, 333]   rank1 [111, 222, 333]   rank2 [111, 222, 333]
```

Done. Every rank has `[111, 222, 333]`. (This trace was generated by simulating the algorithm in Python.)

Why this is a good algorithm:

- **Every link is busy at every step**, each rank sending one chunk and receiving one. No rank is a hotspot.
- Each rank sends `2(N−1)` chunks of size `S/N`, so `2(N−1)/N × S` bytes in total. As N grows this approaches **2S, independent of the number of ranks**. Adding GPUs does not increase the bytes per GPU.

Worked example: data-parallel all-reduce of bf16 gradients for a 7B-parameter model, S = 7e9 × 2 bytes = 14 GB, across N=8 ranks. Each rank sends 2 × 7/8 × 14 = 24.5 GB. At an achieved 50 GB/s per direction (one 400 Gbps NIC's worth), that takes about 0.49 s per step, before any overlap with compute. That is the kind of arithmetic to do before choosing a parallelism layout.

The catch is **latency**. A ring takes `2(N−1)` sequential steps, so for small messages or very large N the per-step latency dominates. That is why NCCL also uses **tree** algorithms (logarithmic depth) and, on NVSwitch systems, in-switch reduction (NVLS/SHARP). The full cost models (the alpha-beta model, bus bandwidth vs algorithm bandwidth, when trees beat rings, and how NCCL chooses) are in [collective communication internals](../network/collectives-internals.md). The `busbw` numbers you see in nccl-tests and the book's benchmarks are normalized by exactly the `2(N−1)/N` factor derived above, so that they can be compared directly with link bandwidth.

## Putting it together: a debugging walk-through

Your 2-node job hangs at startup. Using only this chapter, work from the bottom of the stack up:

1. **Names:** `getent hosts $MASTER_ADDR` on node 1. Is it the right, non-loopback IP?
2. **Route:** `ip route get <master-ip>`. Which interface does it go out of?
3. **Reachability:** `ping <master-ip>` (if ICMP is allowed), then `nc -zv <master-ip> 29500` while the job is starting. Refused, timeout or open?
4. **Listener:** on the master, `ss -tlnp | grep 29500`. Is it listening, and on which address?
5. **Firewall:** host rules, cloud security groups. Are *all* ports allowed between nodes, not just 29500?
6. **NCCL:** rerun with `NCCL_DEBUG=INFO`. Which interface did it pick? Which `NET/` transport? Set `NCCL_SOCKET_IFNAME` if needed.
7. **Fabric:** `ibstat` (ports Active? correct rate?), then `ib_write_bw` between the two nodes, then an all-reduce benchmark.

Most hangs are found by step 5. Most slowdowns are found at step 6.

## You can now read

- [Inter-node and intra-node networking hardware](../network/README.md). The glossary, unidirectional vs bidirectional, frontend/backend/out-of-band networks, PCIe/NVLink/NVSwitch tables, InfiniBand/RoCE/EFA specs. You now know what each row means.
- [Communication patterns](../network/comms.md). The collective diagrams and the ring/tree algorithms.
- [Network debug](../network/debug/README.md). The NCCL debug env vars (`NCCL_DEBUG`, `NCCL_SOCKET_IFNAME`, `NCCL_IB_HCA`, `NCCL_P2P_DISABLE`), P2P checks, Docker gotchas.
- [Network benchmarks](../network/benchmarks/README.md). How `busbw` is computed, latency vs message size, NCCL performance knobs, InfiniBand adaptive routing.
- [Collective communication internals and cost models](../network/collectives-internals.md). The quantitative treatment of everything in the last section.

## Further reading

- James Kurose and Keith Ross, *Computer Networking: A Top-Down Approach*. The standard textbook. Read the chapters on the transport and network layers.
- Brian "Beej" Hall, *Beej's Guide to Network Programming*, https://beej.us/guide/bgnet/. Sockets from C, free and very readable.
- W. Richard Stevens, *TCP/IP Illustrated, Volume 1*. The deep reference for how TCP/IP actually behaves on the wire.
- NVIDIA networking documentation (InfiniBand, ConnectX, UFM, OpenSM), https://docs.nvidia.com/networking/. Also the *RDMA Aware Networks Programming User Manual* for verbs.
- NCCL documentation, environment variables: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html
- PyTorch distributed docs: `torch.distributed`, `torchrun` and rendezvous, https://pytorch.org/docs/stable/distributed.html
- AWS EFA documentation: https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa.html
- Al-Fares, Loukissas, Vahdat, *A Scalable, Commodity Data Center Network Architecture* (SIGCOMM 2008). The fat-tree paper.
- Patarasuk and Yuan, *Bandwidth Optimal All-reduce Algorithms for Clusters of Workstations* (2009). The ring all-reduce bandwidth argument.

# SLURM Administration


## Run a command on multiple nodes

1. to avoid being prompted with:
```
Are you sure you want to continue connecting (yes/no/[fingerprint])?
```
for every new node you haven't logged into yet (ssh asks you to confirm each new host's key, see [ssh: keys, config, tunnels](../../foundations/networking.md#ssh-keys-config-tunnels)), you can disable this check with:
```bash
echo "Host *" >> ~/.ssh/config
echo "  StrictHostKeyChecking no" >> ~/.ssh/config
```

Of course, check if that's secure enough for your needs. I'm making an assumption that you're already on the SLURM cluster and you're not ssh'ing outside of your cluster. You can choose not to set this and then you will have to manually approve each new node.

2. Install `pdsh`

You can now run the wanted command on multiple nodes. `pdsh` runs one command on many hosts in parallel; `PDSH_RCMD_TYPE=ssh` tells it to connect over ssh (rather than its older default, `rsh`), and `-w` takes a Slurm-style host list, where `node-[21,23-26]` means `node-21`, `node-23`, `node-24`, `node-25` and `node-26`. Each output line is prefixed with the host it came from, and they arrive in whatever order the hosts answer.

For example, let's run `date`:

```bash
$ PDSH_RCMD_TYPE=ssh pdsh -w node-[21,23-26] date
node-25: Sat Oct 14 02:10:01 UTC 2023
node-21: Sat Oct 14 02:10:02 UTC 2023
node-23: Sat Oct 14 02:10:02 UTC 2023
node-24: Sat Oct 14 02:10:02 UTC 2023
node-26: Sat Oct 14 02:10:02 UTC 2023
```

Let's do something more useful and complex. Let's kill all GPU-tied processes that didn't exit when the SLURM job was canceled:

First, this command will give us all process ids that tie up the GPUs:

```bash
nvidia-smi --query-compute-apps=pid --format=csv,noheader | sort | uniq
```

(`sort | uniq` removes duplicate PIDs.) So we can now kill all those processes in one swoop - `xargs -n1` runs `sudo kill -9` once per PID, and `-9` is SIGKILL, which the process cannot catch:

```bash
 PDSH_RCMD_TYPE=ssh pdsh -w node-[21,23-26]  "nvidia-smi --query-compute-apps=pid --format=csv,noheader | sort | uniq | xargs -n1 sudo kill -9"
```


## SLURM settings

Show the slurm settings:

```bash
sudo scontrol show config
```

The config file is `/etc/slurm/slurm.conf` on the slurm controller node.

Once `slurm.conf` was updated to reload the config run:
```bash
sudo scontrol reconfigure
```
from the controller node.



## Node management

### Changing the state of the node

The change is performed by `scontrol update`

Examples:

To undrain a node that is ready to be used:
```bash
scontrol update nodename=node-5 state=idle
```

To remove a node from the SLURM's pool:
```bash
scontrol update nodename=node-5 state=drain reason="maintenance"
```


### Undrain nodes killed due to slow process exit

Sometimes processes are slow to exit when a job has been canceled. If the SLURM was configured not to wait forever it'll automatically drain such nodes. But there is no reason for those nodes to not be available to the users.

So here is how to automate it.

The key is to get the list of nodes that are drained due to `"Kill task failed"`, which is retrieved with:

```bash
sinfo -R | grep "Kill task failed"
```

(`sinfo -R` lists the nodes that are down or drained, together with the reason recorded for each, and the node list in its last column.)

now extract and expand the list of nodes (`scontrol show hostnames node-[1-3]` turns a compact list into one hostname per line), check that the nodes are indeed user-process free (or try to kill them first) and then undrain them.

Earlier you learned how to [run a command on multiple nodes](#run-a-command-on-multiple-nodes) which we will use in this script.

Here is the script that does all that work for you: [undrain-good-nodes.sh](./undrain-good-nodes.sh)

Now you can just run this script and any nodes that are basically ready to serve but are currently drained will be switched to `idle` state and become available for the users to be used.


### Auto-reboot

If the nodes need to be rebooted safely (e.g. if the image has been updated), adapt the list of the node and run:

```bash
scontrol reboot ASAP node-[1-64]
```

For each of the non-idle nodes this command will wait till the current job ends, then reboot the node and bring it back up to `idle`.

Note that you need to have:
```
RebootProgram = "/sbin/reboot"
```
set in `/etc/slurm/slurm.conf` on the controller node for this to work (and reconfigure the SLURM daemon if you have just added this entry to the config file).


## Job management

### Modify a job's timelimit

`TimeLimit` takes `days-hours:minutes:seconds`. `$SLURM_JOB_ID` is only set inside the job itself; from outside it, put the job ID shown by `squeue` there instead.

To set a new timelimit on a job, e.g., 2 days:
```bash
scontrol update JobID=$SLURM_JOB_ID TimeLimit=2-00:00:00
```

To add additional time to the previous setting, e.g. 10 more hours.
```bash
scontrol update JobID=$SLURM_JOB_ID TimeLimit=+10:00:00
```

## When something goes wrong with SLURM

Analyze the events log in the SLURM's log file:
```bash
sudo cat /var/log/slurm/slurmctld.log
```

This, for example, can help to understand why a certain node got its jobs canceled before time or the node got removed completely.

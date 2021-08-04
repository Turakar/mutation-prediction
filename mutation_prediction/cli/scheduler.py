import argparse
import asyncio
import os
from asyncio import Future


async def main():
    # argument parsing
    parser = argparse.ArgumentParser(
        description="Schedule single-threaded commands on CPUs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("file", help="A file containing one shell command per line")
    args = parser.parse_args()

    # task list
    cpus = {}

    # kick-start
    for i in os.sched_getaffinity(0):
        cmd = pop_command(args.file)
        if cmd is None:
            break
        task = asyncio.create_task(run(cmd, i))
        cpus[task] = i

    # evaluate until done
    while len(cpus) > 0:
        done, _ = await asyncio.wait(cpus.keys(), return_when="FIRST_COMPLETED")
        task: Future
        for task in done:
            if task.exception():
                raise task.exception()
            cpu = cpus.pop(task)
            cmd = pop_command(args.file)
            if cmd is None:
                print("CPU %d is done." % cpu)
            else:
                task = asyncio.create_task(run(cmd, cpu))
                cpus[task] = cpu
    print("Scheduler done.")


def pop_command(file):
    with open(file, "r") as fd:
        lines = [line for line in fd.readlines() if not line.strip() == ""]
    if len(lines) == 0:
        return None
    cmd = lines[0]
    with open(file, "w") as fd:
        fd.writelines(lines[1:])
    return cmd


async def run(cmd: str, cpu: int):
    print(cmd)
    proc = await asyncio.create_subprocess_shell(
        "taskset -c " + str(cpu) + " " + cmd, stdout=None, stderr=None
    )
    await proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(
            "Process exited with non-zero exit code %d!\n%s" % (proc.returncode, cmd)
        )


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
    loop.close()

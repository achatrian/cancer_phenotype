import argparse
import re
from pathlib import Path
from datetime import datetime
import subprocess  # used to run bash commands from python
import xmltodict

r"""Kill sun grid engine jobs via by name or other quality"""
# TODO embed in bash script
# TODO call qstat in bash script, without saving file to disc, and read into python


if __name__ == '__main__':
    datetime_pattern = '%Y-%m-%dT%H:%M:%S'
    def datetime_obj(string):
        if string == 'now':
            return datetime.now()
        try:
            d = datetime.strptime(string, datetime_pattern)
        except ValueError:
            raise ValueError(f"Invalid datetime string {string}. String must match the following format: 'yyyy-mm-ddThh:MM:SS' (e.g. '2019-09-12T18:57:04')")
        # TODO make this work even when parts are missing (e.g. seconds and minutes)
        return d
    parser = argparse.ArgumentParser()
    parser.add_argument('qstat_file', type=Path)
    parser.add_argument('--name', type=str, default=None, help="Filter jobs to kill by given name")
    parser.add_argument('--name_strict_match', action='store_true', help="Matches whole name rather than just the start")
    parser.add_argument('--state', type=str, default=None, choices=['r', 'qw', 't', 'E', 'T'], help="Filter jobs to kill by their current states")
    parser.add_argument('--queue_name', type=str, default=None, help="Filter jobs to kill by what queue they are running on")
    parser.add_argument('--date_time', type=datetime_obj, default=None, help="'now' for current time, or string describing date 'yyyy-mm-ddThh:MM:SS' : e.g. '2019-09-12T18:57:04', where T separates the date and time of day")
    parser.add_argument('--time_lookup', type=str, choices=['b', 'f'], default='b', help="Filter jobs before (b) or after (forward - f) the given date")
    parser.add_argument('--time_window', type=float, default=None, help="Filter job in given time window")
    args = parser.parse_args()
    with open(args.qstat_file, 'r') as qstat_file:
        jobs = xmltodict.parse(qstat_file.read())['job_info']['job_info']['job_list']
    jobs_to_kill = []
    print(f"Parsing {len(jobs)} jobs ...")
    open(Path('~/jbs_to_kill').expanduser(), 'w').close()  # clear contents of jobs list file
    for job in jobs:
        # run through filters (NB negative logic)
        if args.name is not None:
            if not args.name_strict_match and not re.search(args.name, job['JB_name']):
                continue
            elif args.name_strict_match and not args.name == job['JB_name']:
                continue
        if args.state is not None:
            if not job['state'] == args.state:
                continue
        if args.queue_name is not None:
            if not job['queue_name'] == args.queue_name:
                continue
        if args.date_time is not None:
            # below: cut milliseconds (date string must match format exactly)
            job_date_time = datetime.strptime(job['JB_submission_time'][:-4], datetime_pattern)
            time_delta = (args.date_time - job_date_time).total_seconds()
            if args.time_lookup == 'b':  # kill jobs after given time
                if args.time_window is not None:
                    if time_delta < 0:  #
                        continue
                    elif abs(time_delta) > args.time_window:
                        continue
                elif time_delta < 0:
                    continue
            else:
                if args.time_window is not None:
                    if time_delta > 0:  #
                        continue
                    elif abs(time_delta) > args.time_window:
                        continue
                elif time_delta > 0:
                    continue
        jobs_to_kill.append(job)
        # execute qdel through subprocess bash hook
        # process = subprocess.Popen(f"qdel {job['JB_job_number']}")
        # output, error = process.communicate()
        print(f"Killing job {job['JB_job_number']}")
        with open(Path('~/jbs_to_kill').expanduser(), 'a+') as jobs_to_kill_file:
            jobs_to_kill_file.write(f"{job['JB_job_number']}\n")

# must then use:
#while read i ; do qdel $i ; done < ~/jbs_to_kill













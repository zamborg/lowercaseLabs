import argparse
import json
from pathlib import Path
from .common import atomic_json, read_json
from .runner import Experiment, replay, export_view, freeze


def main():
    p = argparse.ArgumentParser(description='Offline-first reproducible agent-game harness')
    sub = p.add_subparsers(dest='cmd', required=True)
    run = sub.add_parser('run'); run.add_argument('--config'); run.add_argument('--out', required=True); run.add_argument('--bootstrap', type=int, default=2000)
    rep = sub.add_parser('report'); rep.add_argument('directory'); rep.add_argument('--bootstrap', type=int, default=2000)
    replay_cmd = sub.add_parser('replay'); replay_cmd.add_argument('directory')
    view = sub.add_parser('view'); view.add_argument('directory'); view.add_argument('--seat', type=int); view.add_argument('--omniscient', action='store_true'); view.add_argument('--out')
    plan = sub.add_parser('plan'); plan.add_argument('--config', required=True); plan.add_argument('--out', required=True)
    args = p.parse_args()
    if args.cmd == 'plan':
        atomic_json(args.out, freeze(read_json(args.config))); print(args.out)
    elif args.cmd == 'run':
        e = Experiment(args.out, read_json(args.config) if args.config else None)
        records = e.run()
        from .report import write_report
        path = write_report(e.root, records, e.manifest, bootstrap=args.bootstrap)
        print(json.dumps(dict(directory=str(e.root), completed=sum(x['status']=='complete' for x in records), total=len(records), report=str(path))))
    elif args.cmd == 'report':
        from .report import write_report
        directory = Path(args.directory)
        print(write_report(directory, read_json(directory / 'results.json'), read_json(directory / 'manifest.json'), bootstrap=args.bootstrap))
    elif args.cmd == 'replay':
        print(json.dumps(replay(args.directory)))
    else:
        result = export_view(args.directory, args.seat, args.omniscient)
        if args.out:
            atomic_json(args.out, result)
        else:
            print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

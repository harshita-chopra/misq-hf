def get_task(args):
    if args.task == '20q':
        from misq.tasks.twenty_question import Q20Task
        return Q20Task(args)
    elif args.task == 'md':
        from misq.tasks.medical_diagnosis import MDTask
        return MDTask(args)
    elif args.task == 'tb':
        from misq.tasks.troubleshooting import TBTask
        return TBTask(args)
    elif args.task == 'sp':
        from misq.tasks.sp import SPTask
        return SPTask(args)
    else:
        raise NotImplementedError

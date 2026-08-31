from enum import IntFlag, auto

class TaskStatus(IntFlag):
    WAITING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()

    ENQUEUED = auto()
    LATE = auto()

    CRASHED = auto()

TaskStatusTranslation = {
        TaskStatus.WAITING: "Waiting",
        TaskStatus.RUNNING: "Running",
        TaskStatus.COMPLETED: "Completed",
        TaskStatus.FAILED: "Failed",
        TaskStatus.ENQUEUED: "Enqueued",
        TaskStatus.LATE: "Late",
        TaskStatus.CRASHED: "Crashed",
    }

def status_to_string(status: TaskStatus):
    if status in TaskStatusTranslation:
        return TaskStatusTranslation[status]
    res = ""
    for _status in TaskStatus.__iter__():
        if status & _status:
            if _status in TaskStatusTranslation:
                res += TaskStatusTranslation[_status]
    return res

def status_code_to_status(code: int):
    res_status = None
    for status in TaskStatus.__iter__():
        if status & code:
            if not res_status:
                res_status = status
            res_status |= status
    return res_status


def status_code_to_status_string(code: int):
    return status_to_string(status_code_to_status(code))
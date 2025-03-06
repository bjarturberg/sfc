import json
from typing import List

from sfc import Operation, Part, Item, WorkCenter, Counter

def load_gantt_data(path):
    with open(path, 'r') as fp:
        json_data = json.load(fp)

    items = []
    for item in json_data['items']:
        parts = []
        for part in item['project_diagram']:
            operations = []
            for op in part['route']:
                operations.append(Operation(**op))
            part['route'] = operations
            parts.append(Part(**part))
        item['project_diagram'] = parts
        items.append(
            Item(**item)
        )
    return items


def load_work_centers(path):
    with open(path, 'r') as fp:
        json_data = json.load(fp)
    
    work_center_data = json_data['work_center_counters']
    work_centers = []
    for r in work_center_data:
        wc = WorkCenter(
            ID=r['work_center_id'],
            queue_counter=Counter(**r['queue_counter']),
            setup_counter=Counter(**r['setup_counter']),
            in_operation_counter=Counter(**r['in_operation_counter'])
        )
        work_centers.append(wc)

    return work_centers


# 1.1.
def gantt_chart(items: List[Item]):
    """Displays a variant of a Gantt chart of the given items.

    Plots one row per part of each item.
    Example:

    | Item1 - Part 1 |  |> op1 <||> op2 >|    |<op3>|
    | Item1 - Part 2 |                |> op1 <|  |< op2 >|
    |      ...       |

    Each operation segment should consist of three parts: setup time, run time, and move time, in that order.
    Color the operation bars by work_center_id.
    
    Note:
    The relationship between the duration fields and the timestamps in Operation is:

    time:   |  queue_time       | batch_size * run_time_per_unit | move_time |
                   | setup_time |
            ^                   ^                                ^
            |                   |                                |
        arrived_at          started_at                      completed_at
    """


# 1.2
def counter_chart(counter: Counter):
    """Displays a stepped line plot of the counts in counter."""


# 1.3
def simulation_chart(items: list[Item], work_centers: list[WorkCenter]):
    """Displays a gantt chart of all items and two counter charts for each work center, one for the queue and the other for processing.

    The charts are stacked vertically and share the x-axis (time axis).
    """


if __name__ == '__main__':
    path = '/content/sfc/h3_gantt_data.json'
    items = load_gantt_data(path)
    work_centers = load_work_centers(path)

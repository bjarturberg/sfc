"""
Module: sfc.py

Description:
    This module implements a discrete event simulation framework for basic shop‐floor control in manufacturing systems.
    It is based on the sub‐chapter "Basic Shop-Floor Control Concepts" from the chapter "Production Activity Control"
    in "Manufacturing Planning and Control for Supply Chain Management" by F. Robert Jacobs.

    The module defines core classes representing the elements of a shop-floor system, including:
      - Operation: An operation on a part (sub-component) of an item happening on a work station.
      - Part: A sub-component of an item.
      - Item: a complete product consisting of one or more parts (sub-components).
      - WorkCenter: a production resource that schedules and processes operations based on various priority rules.
    It also defines a simple discrete event simulator that can simulate a shop floor system.
      - Counter: used for tracking events on the work center (e.g. operations entering or leaving the queue).
      - PriorityQueue and PrioritizedOperation: support for prioritizing and managing queued operations.
      - Event and DiscreteEventSimulator: components that drive the simulation of operations arriving, starting, and departing.

    The simulation logic handles job scheduling, operation timing, work center utilization, and event management,
    allowing evaluation of performance metrics (such as average processing times and job lateness) based on different
    scheduling rules (e.g., FIFO, shortest processing time, and earliest due date).
"""

from dataclasses import dataclass, field
from functools import total_ordering
from heapq import heappush, heappop
import logging
from typing import List, Literal, Tuple, assert_never, Dict, Optional, NamedTuple


TIME_NOT_SET = -1.0
INT_NOT_SET = -1
PRIORITY_NOT_SET = float("inf")
HIGHEST_PRIORITY = -float("inf")

PriorityRules = Literal[
    "first_in_first_out",  #  FIFO, FCFS
    "shortest_processing_time",  # SPT
    "earliest_due_date",  # EDD
    "order_slack",  # ST, slack time
    "slack_per_operation",  # ST/O slack time per operation
    "critical_ratio",  # CR
    "shortest_operation_next",
    "random",
    "least_work_remaining",  #  LWR
    "next_queue",  # NQ
    "least_setup",  # LSU
]

logging.basicConfig(
    level=logging.ERROR,
    format="%(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class Operation:
    '''An operation represents an operation on a Part at a WorkCenter.
    
    Each operation has three fields that indicate the duration of the operation:
      - setup_time: the time required to prepare for this type of operation.
        The operation type is determined by its ID;
      - run_time_per_unit: the time required to perform the operation on one unit;
      - move_time: the time it takes to move the part to the next work station.
    
    Each operation also has three timestamps:
      - arrived_at: records when the part arrives at the work station
        that handles this operation;
      - started_at: records when the operation is started;
      - completed_at: records when the operation is finished.

    The relationship between the duration fields and the timestamps is:

    time:   |  queue_time       | batch_size * run_time_per_unit | move_time |
                   | setup_time |
            ^                   ^                                ^
            |                   |                                |
        arrived_at          started_at                      completed_at
    '''
    name: str
    work_center_id: int
    run_time_per_unit: float
    batch_size: int
    # run_time: float
    setup_time: float = 0.0
    move_time: float = 0.0
    predicted_queue_time: float = 0.0

    # priority will be determined by priority rule
    priority: float = PRIORITY_NOT_SET

    # Timestamps to be set when processed:
    arrived_at: float = TIME_NOT_SET
    started_at: float = TIME_NOT_SET
    completed_at: float = TIME_NOT_SET
    setup_required: bool = True

    ID: int = INT_NOT_SET  # unique identifier

    @property
    def run_time(self) -> float:
        return self.batch_size * self.run_time_per_unit
    
    def is_completed(self) -> bool:
        return self.completed_at != TIME_NOT_SET

    def has_priority(self) -> bool:
        return self.priority != PRIORITY_NOT_SET

    def queue_time(self) -> float:
        return self.started_at - self.arrived_at

    def non_queue_time(self) -> float:
        return self.move_time + self.setup_time + self.run_time

    def cycle_time(self) -> float:
        return self.setup_time + self.run_time
    
    def predicted_total_time(self) -> float:
        return self.predicted_queue_time + self.non_queue_time()

    def completion_time(self) -> float:
        return self.completed_at - self.arrived_at

    def invariant(self) -> bool:
        """This should always return true.
        If it is not, then the operation is in an illegal state.
        """
        arrived = self.arrived_at
        started = self.started_at
        departed = self.completed_at
        assert started == TIME_NOT_SET or arrived <= started, (
            f"Operation {self.name}: {arrived = } > {started = }."
        )
        assert departed == TIME_NOT_SET or started <= departed, (
            f"Operation {self.name}: {departed = } > {started = }."
        )
        actual_time_at_work_center = departed - started
        cycle_time = self.cycle_time()
        assert departed == TIME_NOT_SET or cycle_time == actual_time_at_work_center, (
            f"Operation {self.name}: {actual_time_at_work_center = } != {cycle_time = }."
        )
        return True


@dataclass
class Part:
    '''A Part represents a part or a sub-component of an Item (final product).'''
    route: List[Operation]
    current_operation_index: int = 0
    ID: int = INT_NOT_SET

    def current_operation(self) -> Operation:
        return self.route[self.current_operation_index]

    def operations_left(self) -> list[Operation]:
        return self.route[self.current_operation_index :]

    def predicted_total_time(self) -> float:
        return sum(operation.predicted_total_time() for operation in self.route)

    def is_completed(self) -> bool:
        return all(op.is_completed() for op in self.route)

    def completed_at(self) -> float:
        if not self.is_completed():
            raise ValueError(f"Part {self.ID} is not completed")
        return self.route[-1].completed_at

    def total_cycle_time_left(self) -> float:
        return sum(operation.cycle_time() for operation in self.operations_left())
    
    def non_queue_time(self) -> float:
        return sum(op.non_queue_time() for op in self.route)

    def n_operations_left(self) -> int:
        return len(self.route) - self.current_operation_index

    def total_time_remaining(self) -> float:
        return sum(operation.non_queue_time() for operation in self.operations_left())


@dataclass
class Item:
    '''An Item represent a final product manufactured.'''
    ID: int
    # For simplicity, we assume that the product structure diagram is just one path.
    project_diagram: List[Part]
    due_date: float
    order_release: float = 0.0

    def is_completed(self) -> bool:
        return all(part.is_completed() for part in self.project_diagram)

    def completed_at(self) -> float:
        if not self.is_completed():
            raise ValueError(f"Item {self.ID} is not completed")
        return max(part.completed_at() for part in self.project_diagram)
    
    def non_queue_time(self) -> float:
        return sum(part.non_queue_time() for part in self.project_diagram)

    def lateness(self) -> float:
        if not self.is_completed():
            raise ValueError(f"Item {self.ID} is not completed")
        return max(0, self.completed_at() - self.due_date)

    def get_part_by_idx(self, part_idx: int) -> Part:
        return self.project_diagram[part_idx]

    def current_operation_in_part(self, part_idx: int) -> Operation:
        return self.project_diagram[part_idx].current_operation()

    def next_part_idx(self, part_idx: int) -> int:
        part = self.get_part_by_idx(part_idx)
        if part.is_completed():
            # All operations in this part are completed, move on to next part.
            return part_idx + 1
        # There is some operation in this part left.
        return part_idx

    def has_next_operation(self, part_idx: int) -> bool:
        """Check whether this part of the item has any operation left."""
        next_part_idx = self.next_part_idx(part_idx)
        return next_part_idx < len(self.project_diagram)


@dataclass
class WorkCenter:
    """A WorkCenter represents some part of the shop floor where an Operation on a Part happens.
    
    Only one part can be worked on at each time.
    All other parts at the WorkStation are in its queue.
    Parts are picked from the queue according to its priority_rule.
    """
    ID: int
    priority_rule: PriorityRules = "first_in_first_out"
    queue: "PriorityQueue" = field(default_factory=lambda: PriorityQueue())
    is_busy: bool = False
    in_operation_counter: "Counter" = field(default_factory=lambda: Counter())
    queue_counter: "Counter" = field(default_factory=lambda: Counter())
    setup_counter: "Counter" = field(default_factory=lambda: Counter())
    current_setup_id: int = INT_NOT_SET

    def next_job(self) -> Optional["PrioritizedOperation"]:
        if self.queue.is_empty():
            return
        return self.queue.pop()

    def determine_priority(
        self, operation: Operation, part_idx: int, item: Item
    ) -> float:
        priority_rule = self.priority_rule
        if priority_rule == "first_in_first_out":
            return operation.arrived_at
        if priority_rule == "shortest_processing_time":
            return operation.run_time
        if priority_rule == "earliest_due_date":
            return item.due_date
        if priority_rule == "order_slack":
            raise NotImplementedError
        if priority_rule == "slack_per_operation":
            raise NotImplementedError
        if priority_rule == "critical_ratio":
            raise NotImplementedError
        if priority_rule == "shortest_operation_next":
            raise NotImplementedError
        if priority_rule == "random":
            raise NotImplementedError
        if priority_rule == "least_work_remaining":
            raise NotImplementedError
        if priority_rule == "next_queue":
            raise NotImplementedError
        if priority_rule == "least_setup":
            raise NotImplementedError
        assert_never(priority_rule)
        raise ValueError(f"Unknown priority rule {self.priority_rule}")


###############################################################################
############################### SIMULATION CODE ###############################
###############################################################################


@dataclass
class Counter:
    data: List[Tuple[float, int, str]] = field(
        default_factory=lambda: [(-1.0, 0, "start")]
    )

    def up(self, time: float, message: str = "up"):
        data = self.data
        current_count = data[-1][1]
        data.append((time, current_count + 1, message))

    def down(self, time: float, message: str = "down"):
        data = self.data
        current_count = data[-1][1]
        data.append((time, current_count - 1, message))

    def zipped(self) -> List[Tuple[float, int, str]]:
        return self.data


class PrioritizedOperation(NamedTuple):
    """This is used to add operations to the work center queues.

    A PrioritizedOperation object, obj, is the ordered by obj.priority.
    obj.priority willl usually be equal to obj.operation.priority.
    """

    operation: Operation
    part_idx: int
    item: Item
    priority: float

    def __lt__(self, other: "PrioritizedOperation") -> bool:  # type: ignore
        if not isinstance(other, PrioritizedOperation):
            return NotImplemented
        return self.priority < other.priority

    def __le__(self, other: "PrioritizedOperation") -> bool:  # type: ignore
        if not isinstance(other, PrioritizedOperation):
            return NotImplemented
        return self.priority <= other.priority

    def __gt__(self, other: "PrioritizedOperation") -> bool:  # type: ignore
        if not isinstance(other, PrioritizedOperation):
            return NotImplemented
        return self.priority > other.priority

    def __ge__(self, other: "PrioritizedOperation") -> bool:  # type: ignore
        if not isinstance(other, PrioritizedOperation):
            return NotImplemented
        return self.priority >= other.priority


@dataclass
class PriorityQueue:
    queue: List[PrioritizedOperation] = field(default_factory=list)

    def put(self, operation: Operation, part_idx: int, item: Item):
        assert operation.has_priority(), (
            "Operation cannot be added to queue with no priority"
        )
        logger.debug(f"Pushing {operation.name}, priority = {operation.priority}")
        heappush(
            self.queue,
            PrioritizedOperation(operation, part_idx, item, operation.priority),
        )

    def pop(self) -> PrioritizedOperation:
        return heappop(self.queue)

    def is_empty(self) -> bool:
        return len(self.queue) == 0


@total_ordering
class Event(NamedTuple):
    type: Literal["arrive", "depart"]
    time: float
    work_center: "WorkCenter"  # type: ignore
    item: "Item"  # type: ignore
    part_idx: int

    # Priority is just used in commparison when time is equal.
    priority: float = HIGHEST_PRIORITY

    def part(self) -> "Part":  # type: ignore
        return self.item.project_diagram[self.part_idx]

    def log(self) -> str:
        current_operation = self.item.get_part_by_idx(self.part_idx).current_operation()
        return f"{self.time} {self.type}: {current_operation.name}"

    def operation(self) -> "Operation":
        return self.item.current_operation_in_part(self.part_idx)

    def __lt__(self, other: "Event"):  # type: ignore
        if not isinstance(other, Event):
            return NotImplemented
        return (self.time, self.priority) < (other.time, other.priority)

    def __eq__(self, other: "Event"):  # type: ignore
        if not isinstance(other, Event):
            return NotImplemented
        return (self.time, self.priority) == (other.time, other.priority)


class DiscreteEventSimulator:
    def __init__(self, work_centers: List[WorkCenter], items: List[Item]):
        self.event_queue: List[Event] = []  # Priority queue for events
        self.time: float = -1.0  # Simulation clock
        self.work_centers_by_id: Dict[int, WorkCenter] = {
            wc.ID: wc for wc in work_centers
        }
        self.items = items
        self.setup(items)

    def setup(self, items: List[Item]):
        logger.info("Set up simulator.")
        for item in self.items:
            part_idx = 0
            part = item.get_part_by_idx(part_idx)
            first_operation = part.current_operation()
            work_center = self.work_centers_by_id[first_operation.work_center_id]

            event = Event(
                type="arrive",
                work_center=work_center,
                item=item,
                part_idx=part_idx,
                time=item.order_release,
                priority=work_center.determine_priority(
                    first_operation, part_idx, item
                ),
            )
            self.schedule_event(event)

    def schedule_event(self, event: Event):
        heappush(self.event_queue, event)

    def run(self, until):
        logger.info("Start running simulation.")
        time = self.time
        while self.event_queue and time < until:
            event = heappop(self.event_queue)
            if event.type == "arrive":
                logger.debug(event.log())
                self.handle_arrival(event)
            elif event.type == "depart":
                logger.debug(event.log())
                self.handle_departure(event)
            else:
                assert_never(event.type)
                raise ValueError(f"Unknown event type {event.type}")
            time = event.time

    def handle_arrival(self, event: Event):
        event_type, time, work_center, item, part_idx, priority = event
        operation = event.operation()
        operation.arrived_at = time
        work_center.queue_counter.up(time, f"added to queue {operation.name}")
        operation.priority = work_center.determine_priority(operation, part_idx, item)
        if work_center.is_busy:
            work_center.queue.put(operation, part_idx, item)
        else:
            self.start(operation, part_idx, item, work_center, time)

    def start(
        self,
        operation: Operation,
        part_idx: int,
        item: Item,
        work_center: WorkCenter,
        time: float,
    ):
        assert not work_center.is_busy, (
            f"Trying to start {operation.name} when work center {work_center.ID} is busy"
        )
        logger.debug(f"{time} start: {operation.name}")
        # Update states:
        work_center.setup_counter.up(time)
        if work_center.current_setup_id == operation.ID:
            start_time = time
            operation.setup_required = False
        else:
            start_time = time + operation.setup_time
            operation.setup_required = True
        work_center.current_setup_id = operation.ID
        work_center.setup_counter.down(start_time)
        operation.started_at = start_time
        work_center.queue_counter.down(start_time, f"left queue {operation.name}")
        work_center.in_operation_counter.up(start_time, f"start {operation.name}")
        work_center.is_busy = True
        # Schedule depart event:
        depart_event = Event(
            type="depart",
            work_center=work_center,
            item=item,
            part_idx=part_idx,
            time=start_time + operation.run_time,
            priority=HIGHEST_PRIORITY,
        )
        self.schedule_event(depart_event)

    def handle_departure(self, event: Event):
        event_type, time, work_center, item, part_idx, priority = event
        departing_operation = event.operation()
        part = event.part()

        # Update operation, part and work_center state.
        # Item state is updated below.
        departing_operation.completed_at = time
        part.current_operation_index += 1
        work_center.is_busy = False
        work_center.in_operation_counter.down(
            event.time, f"depart {departing_operation.name}"
        )

        # Handle any queued operations:
        next_job = work_center.next_job()
        if next_job is not None:
            self.start(
                next_job.operation, next_job.part_idx, next_job.item, work_center, time
            )

        # Update item state:
        if not item.has_next_operation(part_idx):
            return
        # There are more operations to schedule
        next_part_idx = item.next_part_idx(part_idx)
        next_operation = item.get_part_by_idx(next_part_idx).current_operation()
        next_work_center = self.work_centers_by_id[next_operation.work_center_id]

        # We assume that next_operation cannot start until the current part
        # has been moved.
        # Would this necessarily be true if the next operation does not
        # belong to the same part as the departing operation?
        next_arrival_time = time + departing_operation.move_time
        next_arrival = Event(
            type="arrive",
            item=item,
            work_center=next_work_center,
            part_idx=next_part_idx,
            time=next_arrival_time,
            priority=HIGHEST_PRIORITY,
        )
        self.schedule_event(next_arrival)


# HELPERS:
def tabulate(table: List[List[int | float | str]], column_names: List[str]):
    """
    Prints a formatted table.

    Args:
        table (list[list[int | float | str]]): 2D list representing table rows.
        column_names (list[str]): List of column names.
    """
    # Determine the number of columns
    n_cols = len(column_names)

    # Compute maximum width for each column based on header and row values
    col_widths = [len(name) for name in column_names]
    for row in table:
        for i in range(n_cols):
            col_widths[i] = max(col_widths[i], len(str(row[i])))

    # Build a row format string based on the computed column widths
    row_format = " | ".join(f"{{:<{width}}}" for width in col_widths)

    # Print header row
    print(row_format.format(*column_names))
    # Print separator
    print("-+-".join("-" * width for width in col_widths))
    # Print each row in the table
    for row in table:
        print(row_format.format(*(str(cell) for cell in row)))

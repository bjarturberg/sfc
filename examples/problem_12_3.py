"""
The Ace Tool Company is considering implementing the repetitive lot concept in scheduling the firm’s fabrication shop. The production manager selected an example order to use in evaluating benefits and potential costs of this scheduling approach. A transfer batch size of 100 units was suggested for this item. The example order is for a quantity of 1,000 units and has the following routing data:

Operation  |  Work Center  |  Setup Time  |  Run Time/Unit  |
    1             1               40m             2.4m
    2             2               20m             1.44m


Assuming a single-shift, eight-hour day, five-day week for work centers 1 and 2, prepare a Gantt chart showing the earliest start- and finish-time schedule for this order under a conventional scheduling approach where all items in the order are processed at one time. Do the same when the repetitive lot concept is used. What are the earliest start and finish times for each transfer batch at work center 2, assuming none of the transfer batches are processed together to save setup time?
What is the difference in the order-completion times under the two scheduling approaches in part a above?
What are the benefits and potential costs of this scheduling approach?

"""

from sfc import Operation, Part, Item, WorkCenter, DiscreteEventSimulator, tabulate


def conventional_schedule():
    wc1 = WorkCenter(ID=1, priority_rule="first_in_first_out")
    wc2 = WorkCenter(ID=2, priority_rule="first_in_first_out")

    op1 = Operation(
        name="1",
        work_center_id=wc1.ID,
        run_time_per_unit=2.4,
        batch_size=1000,
        setup_time=40.0,
        ID=1
    )
    op2 = Operation(
        name="2",
        work_center_id=wc2.ID,
        run_time_per_unit=1.44,
        batch_size=1000,
        setup_time=20.0,
        ID=2
    )
    items = [
        Item(
            ID=1,
            project_diagram=[
                Part(route=[op1, op2])],
            due_date=10000
        )
    ]
    simulation = DiscreteEventSimulator(work_centers=[wc1, wc2], items=items)
    simulation.run(until=1e10)
    for item in items:
        print(item.completed_at())


def repetitive_lot():
    wc1 = WorkCenter(ID=1, priority_rule="first_in_first_out")
    wc2 = WorkCenter(ID=2, priority_rule="first_in_first_out")
    total_batch_size = 1000
    n_operation_batches = 10
    operation_batch_size = total_batch_size / n_operation_batches

    def op1():
        return Operation(
            name="1",
            work_center_id=wc1.ID,
            run_time_per_unit=2.4,
            batch_size=operation_batch_size,
            setup_time=40.0,
            ID=1,
        )

    def op2():
        return Operation(
            name="2",
            work_center_id=wc2.ID,
            run_time_per_unit=1.44,
            batch_size=operation_batch_size,
            setup_time=20.0,
            ID=2,
        )

    items = [
        Item(
            ID=i,
            project_diagram=[
                Part(route=[op1(), op2()])
            ],
            due_date=10000
        )
        for i in range(n_operation_batches)
    ]
    simulation = DiscreteEventSimulator(work_centers=[wc1, wc2], items=items)
    simulation.run(until=1e10)
    sorted_items = sorted(items, key=lambda i: i.completed_at())
    table = []
    for item in sorted_items:
        o1, o2 = item.project_diagram[0].route
        table.append(
            [
                o1.setup_required,
                o1.started_at,
                o1.completed_at,
                o2.setup_required,
                o2.started_at,
                o2.completed_at,
            ]
        )

    tabulate(table, ["setup 1", "start 1", "end 1", "setup 2", "start 2", "end 2"])

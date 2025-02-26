from sfc import Operation, Part, Item, WorkCenter, DiscreteEventSimulator, tabulate


# You are given the following data on jobs.
# Each row shows (name, processing time, due date) for a job.
data = [
    ("A", 4, 8),
    ("B", 13, 37),
    ("C", 6, 8),
    ("D", 3, 7),
    ("E", 11, 39),
    ("F", 9, 21),
    ("G", 8, 16),
]


work_center = WorkCenter(ID=1, priority_rule="shortest_processing_time")

items = [
    Item(
        ID=1,
        project_diagram=[
            Part(
                route=[
                    Operation(
                        name=name,
                        work_center_id=work_center.ID,
                        run_time=float(run_time),
                    )
                ]
            )
        ],
        due_date=due_date,
    )
    for i, (name, run_time, due_date) in enumerate(data)
]

print(
    "a. Using the shortest processing time scheduling rule, in what order would the jobs be completed? Processing can start immediately."
)
simulation = DiscreteEventSimulator(work_centers=[work_center], items=items)
simulation.run(until=1000.0)
operations = [(i.project_diagram[0].route[0], i) for i in items]
sorted_operations = sorted(operations, key=lambda x: x[0].started_at)
results = [
    [
        op.name,
        op.started_at,
        op.run_time,
        op.completed_at,
        item.due_date,
        item.lateness(),
    ]
    for op, item in sorted_operations
]
tabulate(results, ["job", "started", "run_time", "completed", "due_date", "lateness"])


print(
    "# b. What is the average completion time (in days) of the sequence calculated in part a?"
)
print(sum(op.completion_time() for op, _ in operations) / len(operations))

print(
    "c. What is the average job lateness (in days) of the sequence calculated in part a?"
)
print(sum(item.lateness() for item in items) / len(items))

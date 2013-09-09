def find_boundary(edges):
    """Find input and output nodes from a graph defined by edges"""

    inputs = set([x[0] for x in edges])
    outputs = set([x[1] for x in edges])
    for e in edges:
        inputs.discard(e[1])
        outputs.discard(e[0])
    return inputs, outputs

def find_order(edges):
    order = []
    (inputs, _) = find_boundary(edges)
    edges_covered = []

    queue = list(inputs)
    while queue != []:
        name = queue.pop(0)
        add_it = True
        for e in edges:
            if e not in edges_covered:
                if e[0] == name:
                    if e[1] not in queue:
                        queue += [e[1]]
                    edges_covered += [e]
                elif e[1] == name:
                    add_it = False
        if add_it:
            order += [name]

    return order

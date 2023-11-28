def PlotPath(paths, nodes, connections, start, end):
    # Check how many paths reached the end:
    counter = 0
    completedPaths = []
    for i,path in enumerate(paths):
        if path[-1] == end:
            counter += 1
            completedPaths.append(i)

    figure, axs = plt.subplots(1, counter)
    x = [node[0] for node in nodes]
    y = [node[1] for node in nodes]

    for k, ax in enumerate(axs):
        # Find and plot ALL connections
        pairs = []
        pairCoord = []
        for i, connections in enumerate(connections):
            for j, connection in enumerate(connections):
                if connection == 1:
                    pairs.append([i, j])
                    pairCoord.append([[x[i], x[j]], [y[i], y[j]]])
        for pair in pairCoord:
            ax.plot(pair[0], pair[1], linewidth=0.1, color='blue')
    
        # Plot path:
        print(completedPaths)
        print(len(paths), k)
        path = paths[completedPaths[k]]
        print(path)
        xPath = [x[edge] for edge in path]
        yPath = [y[edge] for edge in path]
        ax.plot(xPath, yPath, linewidth=1, color='red')

        # Add nodes
        ax.scatter(x, y, color='black')
        # Highlight start and end nodes
        ax.scatter([x[start], x[end]], [y[start], y[end]], color='green')
        plt.show()
    plt.show()
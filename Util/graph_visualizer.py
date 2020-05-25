import sys
import networkx as nx
from pathlib import Path
import matplotlib.pyplot as plt
#dir_path = os.path.dirname(os.path.realpath(__file__))

def read_graph(path):
    dir_path = Path.cwd()
    parent_path = Path(dir_path)
    parent_path = parent_path.parent
    create_path = parent_path/'Result'
    if not create_path.exists():
        create_path.mkdir(parents=True,exist_ok=True)
        print("\ncreate 'Result' subfolder...... ")

    # use networkx for visualizing
    for i in range(len(path)):
        G = nx.read_graphml(path[i])
        posx = nx.get_node_attributes(G,"posx")
        posy = nx.get_node_attributes(G,"posy")
        pos={}
        # start drawing the graph, first record the pos
        for (k1,v1), (k2,v2) in zip(posx.items(),posy.items()):
            pos[k1]=(v1,v2)
        # labels

        labs = nx.get_node_attributes(G,"label")
        print("\nfigure {} with labels:\n{}".format(i,labs))
        plt.figure(i)
        fig, ax = plt.subplots()
        nx.draw_networkx_nodes(G,pos,node_color=list(labs.values()),ax=ax)
        nx.draw_networkx_edges(G,pos)
        nx.draw_networkx_labels(G,pos,labels=labs)
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    plt.show()

    

if __name__=="__main__":
    if(len(sys.argv)!<2):
        print("please provide at least 1 graphml for drawing")
    paths=[]
    for i in range(len(sys.argv)-1):
        paths.append(sys.argv[i+1])
    read_graph(paths)


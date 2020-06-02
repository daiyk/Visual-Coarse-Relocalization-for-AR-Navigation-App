import sys
import networkx as nx
import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import PIL.Image as Image
#dir_path = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser(description="Graph visualization program")
parser.add_argument("-g",action="append",nargs="+", help="paths to graphs for visulization[required]",required=True)

parser.add_argument("-scl", default = 0.5, type=float, help="the rescale value of image[default: 0.5]")

def read_graph(path,scale):
    dir_path = Path.cwd()
    parent_path = Path(dir_path)
    parent_path = parent_path.parent
    create_path = parent_path/'Result'
    if not create_path.exists():
        create_path.mkdir(parents=True,exist_ok=True)
        print("\ncreate 'Result' subfolder...... ")
    # check whether the 
    # use networkx for visualizing
    for i in range(len(path)):
        filesteam = Path(path[i])
        img1 = filesteam.with_suffix('.jpg')
        img2 = filesteam.with_suffix('.JPG')
        img3 = filesteam.with_suffix('.jpeg')
        if img1.exists():
            tempImg = Image.open(img1)
            tempImg = tempImg.resize((int(tempImg.size[0]*scale),int(tempImg.size[1]*scale)),Image.ANTIALIAS)
            img = np.asarray(tempImg)
        elif img2.exists():
            tempImg = Image.open(img2)
            tempImg = tempImg.resize((int(tempImg.size[0]*scale),int(tempImg.size[1]*scale)),Image.ANTIALIAS)
            img = np.asarray(tempImg)
            # img = mpimg.imread(img2)
        elif img3.exists():
            tempImg = Image.open(img3)
            tempImg = tempImg.resize((int(tempImg.size[0]*scale),int(tempImg.size[1]*scale)),Image.ANTIALIAS)
            img = np.asarray(tempImg)
            # img = mpimg.imread(img3)
        else:
            print("\n please provides corresponding image to the graphml file\n")        
            break
        G = nx.read_graphml(path[i])
        posx = nx.get_node_attributes(G,"posx")
        posy = nx.get_node_attributes(G,"posy")
        pos={}
        # start drawing the graph, first record the pos
        for (k1,v1), (k2,v2) in zip(posx.items(),posy.items()):
            pos[k1]=(v1,v2)
        # labels
        mpimg
        labs = nx.get_node_attributes(G,"label")
        print("\nfigure {} with labels:\n{}".format(i,labs))
        plt.figure(i)
        fig, ax = plt.subplots()

        #resize img
        
        plt.imshow(img)
        nx.draw_networkx_nodes(G,pos,node_color=list(labs.values()),ax=ax)
        nx.draw_networkx_edges(G,pos)
        nx.draw_networkx_labels(G,pos,labels=labs)
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    plt.show()


if __name__=="__main__":
    args = parser.parse_args()
    paths = args.g[0]
    print(paths)
    scale = args.scl
    read_graph(paths,scale)
   


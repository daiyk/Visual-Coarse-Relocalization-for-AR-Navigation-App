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

parser.add_argument("-scl", required=True, type=float, help="the rescale value of image[default: 0.5]")
parser.add_argument("-need_img", required=False,default=False, type=bool, help="show the image in the background[default: True]")


def recurKernelPlot(path,scale,plot_full_set=False,need_img=False):
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
        if(need_img):
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
        h = nx.get_node_attributes(G,"h")
        pos={}
        print(h)
        colorCate=['None','g', 'c', 'm', 'y', 'k', 'w']
        colorMap=[]
        for k,v in h.items():
            if v==10.0:
                colorMap.append('r')
            elif v==0.0:
                colorMap.append('b')
            else:
                colorMap.append(colorCate[int(v)])
        totalEdges = [e for e in G.edges()]
        labs = nx.get_node_attributes(G,"label")
        
        labs_edges={}
        # # start drawing the graph, first record the pos
        for (k1,v1), (k2,v2) in zip(posx.items(),posy.items()):
            pos[k1]=(v1,v2)

        #iterate through the graph and build graph for edges-only nodes
        #G_edges = nx.Graph()
        G_edges = nx.MultiGraph()
        pos_edges={}
        for u,v in G.edges():
            G_edges.add_nodes_from([u,v])
            pos_edges[u]=pos[u]
            pos_edges[v]=pos[v]
            labs_edges[u]=labs[u]
            labs_edges[v]=labs[v]
        # labels
        G_edges.add_edges_from(G.edges)
        mpimg
        
        # plt.figure(i)
        fig, ax = plt.subplots()
        print(" nodes: {}".format(G_edges.number_of_nodes()))
        #resize img
        if(plot_full_set):
            print("{}th edge number: {};".format(i,G_edges.number_of_edges()))
            if(need_img):
                plt.imshow(img)
            p = Path(path[i])    
            plt.title(p.stem)
            nx.draw_networkx_nodes(G,pos,node_color=colorMap,ax=ax,node_size=200)
            nx.draw_networkx_edges(G,pos)
            nx.draw_networkx_labels(G,pos,labels=labs)
            ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        else:
            print("{}th edge number: {}; ".format(i,G_edges.number_of_edges()))
            if(need_img):
                plt.imshow(img)
            p = Path(path[i])    
            plt.title(p.stem)
            nx.draw_networkx_nodes(G_edges,pos_edges,node_color=list(h.values()),ax=ax)
            nx.draw_networkx_edges(G_edges,pos_edges)
            nx.draw_networkx_labels(G_edges,pos_edges,labels=labs_edges)
            ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    plt.show()


def read_graph(path,scale,plot_full_set=False,need_img=False):
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
        if(need_img):
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

        totalEdges = [e for e in G.edges()]
        labs = nx.get_node_attributes(G,"label")
        weight = nx.get_edge_attributes(G,"weight")
        labs_edges={}
        # # start drawing the graph, first record the pos
        for (k1,v1), (k2,v2) in zip(posx.items(),posy.items()):
            pos[k1]=(v1,v2)

        #iterate through the graph and build graph for edges-only nodes
        G_edges = nx.Graph()
        pos_edges={}
        for u,v in G.edges():
            G_edges.add_nodes_from([u,v])
            pos_edges[u]=pos[u]
            pos_edges[v]=pos[v]
            labs_edges[u]=labs[u]
            labs_edges[v]=labs[v]
        # labels
        G_edges.add_edges_from(G.edges)
        mpimg
        
        # plt.figure(i)
        fig, ax = plt.subplots()

        #resize img
        if(plot_full_set):
            print("{}th edge number: {};".format(i,G_edges.number_of_edges()))
            if(need_img):
                plt.imshow(img)
            p = Path(path[i])    
            plt.title(p.stem)
            nx.draw_networkx_nodes(G,pos,node_color=list(labs.values()),ax=ax)
            nx.draw_networkx_edges(G,pos)
            nx.draw_networkx_labels(G,pos,labels=labs)
            ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        else:
            print("{}th edge number: {}; ".format(i,G_edges.number_of_edges()))
            if(need_img):
                plt.imshow(img)
            p = Path(path[i])    
            plt.title(p.stem)
            nx.draw_networkx_nodes(G_edges,pos_edges,node_color=list(labs_edges.values()),ax=ax)
            nx.draw_networkx_edges(G_edges,pos_edges)
            nx.draw_networkx_labels(G_edges,pos_edges,labels=labs_edges)
            nx.draw_networkx_edge_labels(G_edges,pos_edges,edge_labels=weight)
            ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    plt.show()



if __name__=="__main__":
    args = parser.parse_args()
    paths = args.g[0]
    print(paths)
    scale = args.scl
    need_img=args.need_img
    plot_full_set = False #means to delete sets of vertex / node(that without any edge)
    # recurKernelPlot(paths,scale, True,need_img=True)
    read_graph(paths,scale, plot_full_set,need_img=need_img)
  
   


import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser(description="cumulative distribution function visualizing")
parser.add_argument("-d",action="append",nargs="+", help="paths to graphs for visulization[required]",required=True)

def plot_cdf(paths):
    
    scales=[0,1,2,3,4]
    fig = plt.figure()
    for i in range(len(paths)):
        path = paths[i]
        scores=[]
        with open(path,"r") as file:
            reader = csv.reader(file)
            header = next(reader,None)
            for row in reader:
                scores.append(int(row[0]))
            # draw cdf
            scores = np.array(scores)
            scores.astype(int)

            # find the statisics of the scores function
            stat = np.zeros(5,dtype=np.int32)
            for i in scores:
                stat[i]=stat[i]+1
            
            #draw cdf plot
            stat_cum = np.cumsum(stat)/np.sum(stat)
            plt.plot(scales,stat_cum,label=header[0],linewidth=3)
    plt.xlabel("UKB score")
    plt.ylim([0,1])
    plt.ylabel("CDF")
    plt.xticks(scales,scales)
    fig.suptitle("Cumulative Distribution Function")
    plt.legend()
    plt.show()
    
if __name__=="__main__":
    args = parser.parse_args()
    paths = args.d[0]
    print(paths)
    plot_cdf(paths)
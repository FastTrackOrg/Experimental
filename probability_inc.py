'''
    Copyright (C) 2020  FastTrackOrg

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.'''

import scipy
import scipy.integrate as integrate
import scipy.stats
from scipy.optimize import curve_fit
from scipy.spatial import Voronoi
import numpy as np
import shapely
import pandas as pd
from shapely.geometry import LinearRing
from shapely.geometry import LineString
import matplotlib.pyplot as plt
import cv2

def geometricProbability(points, resolution):
    '''
    Computes the geometric probability for one image.
    '''
    density = len(points) / (resolution[0]*resolution[1])
    vor = Voronoi(points)
    probaCell = np.zeros((5000))
    count = 0
    for j, i in enumerate(vor.point_region):
        if not -1 in vor.regions[i] and vor.regions[i]:
            probaCell += probability(vor.points[j], vor.vertices[vor.regions[i]], density)
            count += 1
    return probaCell/count


def probability(center, vertices, density):
    '''
    Computes the geometric probability for one Voronoï cell
    '''
    # Computes the reduced distance between the point and the Voronoï edge. 
    polygon = LinearRing(vertices)
    rmax = np.zeros(360)
    for j, i in enumerate(np.linspace(0, 359*np.pi/180, 360)): # CHECK
        line = LineString([center, [center[0] + 2/np.sqrt(density)*np.cos(i), center[1] + 2/np.sqrt(density)*np.sin(i)]])
        r = line.intersection(polygon)
        try:
            rmax[j] = np.sqrt(density)*np.sqrt((r.x - center[0])**2 + (r.y - center[1])**2)
        except:
            rmax[j] = np.nan
    rmax = rmax[~np.isnan(rmax)]

    # Compute the proportion where a point j is outside the Voronoi cell
    p = np.zeros(5000)
    for j, i in enumerate(np.linspace(0, 1.5, 5000)):
        tmp = 0
        for k in rmax:
            if i > k:
                tmp +=1
        p[j] = tmp/rmax.shape[0]
    return p

def distanceDist(data, resolution, timescale=1):
    '''
    Compute the reduced displacement distribution
    '''
    d = []
    for i in set(data.id.values):
        if not data[(data.id == i) & (data.imageNumber.values%timescale == 0)].empty:
            d.extend((timescale*np.sqrt(np.diff(data[(data.id == i) & (data.imageNumber.values%timescale == 0)].xHead.values)**2 + np.diff(data.yHead[(data.id == i) & (data.imageNumber.values%timescale == 0)].values)**2)/np.diff(data.imageNumber[(data.id == i) & (data.imageNumber.values%timescale == 0)].values))*np.sqrt(len(set(data.id.values)) / (resolution[0]*resolution[1])))
        else:
            d.append(np.sqrt((data[data.id == i].xHead.values[-1]-data[data.id == i].xHead.values[0])**2+(data[data.id == i].yHead.values[-1]-data[data.id == i].yHead.values[0])**2)*np.sqrt(len(set(data.id.values)) / (resolution[0]*resolution[1])))
    dist = scipy.stats.gaussian_kde(d)
    return dist
        
            

def run(path, tau, cells, plot=False):
    data = pd.read_csv(path + "/tracking.txt", sep='\t')

    cells = int(cells/len(set(data.id.values)))
    if cells > np.max(data.imageNumber.values): cells = -1
    if cells == 0: cells = 1
    resolution = cv2.imread(path + "/background.pgm").shape
    pinc = np.zeros(5000)
    for i in set(data.imageNumber.values[0:cells]):
        dat = data[data.imageNumber == i]
        points = list(zip(dat.xHead, dat.yHead))
        pinc += geometricProbability(points, resolution)
    pinc /= len(set(data.imageNumber.values[0:cells]))
    def pincDist(x, xo, L, k): return L/(1+np.exp(-k*(x-xo)))
    popt, __ = curve_fit(pincDist, np.linspace(0, 1.5, 5000), pinc)

    reducedDist = distanceDist(data, resolution, tau)
    probabilityOfIncursion = integrate.quad(lambda x, popt:pincDist(x, *popt)*reducedDist(x), 0, 100, args=popt)
    
    if plot:
        plt.figure()
        x = np.linspace(0, 1.5, 5000)
        plt.plot(x, pincDist(x, *popt), "o", label="pinc")
        plt.xlabel("Reduced distance")
        plt.ylabel("Geometric probability")

        plt.figure()
        x = np.linspace(0, 1.5, 5000)
        plt.xlabel("Timescale")
        plt.ylabel("Probability")
        for i in range(1, np.max(data.imageNumber.values)):
            reducedDist = distanceDist(data, resolution, i)
            probabilityOfIncursionTmp = integrate.quad(lambda x, popt:pincDist(x, *popt)*reducedDist(x), 0, 100, args=popt)
            plt.scatter(i, probabilityOfIncursionTmp[0])
        plt.show()

    print(probabilityOfIncursion[0])
    return probabilityOfIncursion[0]

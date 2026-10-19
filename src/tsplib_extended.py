"""Read the symmetric TSPLIB distance types used in the v6 extension."""
from pathlib import Path
import numpy as np
from .tsplib_parser import TSPInstance, parse_tsp, _split_header


def parse_extended(path):
    path=Path(path)
    lines=path.read_text().splitlines()
    header={}; section=None; coords=[]; numbers=[]
    for raw in lines:
        line=raw.strip()
        if not line: continue
        if line=='EOF': break
        if line.endswith('_SECTION'):
            section=line; continue
        if section=='NODE_COORD_SECTION':
            row=line.split(); coords.append([float(row[1]),float(row[2])])
        elif section=='EDGE_WEIGHT_SECTION':
            numbers.extend(map(int,line.split()))
        elif section is None:
            k,v=_split_header(line);header[k]=v
    n=int(header['DIMENSION']);kind=header['EDGE_WEIGHT_TYPE']
    if kind=='EUC_2D':return parse_tsp(path)
    xy=np.asarray(coords,dtype=float).reshape((-1,2))
    if kind in ('ATT','GEO'):
        if len(xy)!=n:raise ValueError('Coordinate count mismatch')
        if kind=='ATT':
            diff=xy[:,None,:]-xy[None,:,:]
            r=np.sqrt(np.sum(diff*diff,axis=2)/10.)
            t=np.floor(r+.5)
            dist=(t+(t<r)).astype(np.int32)
        else:
            deg=np.trunc(xy);rad=np.pi*(deg+5.*(xy-deg)/3.)/180.
            lat,lon=rad[:,0],rad[:,1]
            q1=np.cos(lon[:,None]-lon[None,:])
            q2=np.cos(lat[:,None]-lat[None,:])
            q3=np.cos(lat[:,None]+lat[None,:])
            angle=.5*((1+q1)*q2-(1-q1)*q3)
            dist=np.floor(6378.388*np.arccos(np.clip(angle,-1.,1.))+1.).astype(np.int32)
    elif kind=='EXPLICIT':
        dist=np.zeros((n,n),dtype=np.int32);fmt=header['EDGE_WEIGHT_FORMAT']
        if fmt=='FULL_MATRIX':dist=np.asarray(numbers,dtype=np.int32).reshape((n,n))
        else:
            diagonal='DIAG' in fmt; upper=fmt.startswith('UPPER')
            if not fmt.endswith('_ROW'):raise NotImplementedError(fmt)
            k=0
            for i in range(n):
                indices=range(i if diagonal else i+1,n) if upper else range(i+1 if diagonal else i)
                for j in indices:
                    dist[i,j]=dist[j,i]=numbers[k];k+=1
            if k!=len(numbers):raise ValueError('Weight count mismatch')
    else:raise NotImplementedError(kind)
    np.fill_diagonal(dist,0)
    if not np.array_equal(dist,dist.T) or np.any(dist<0):raise ValueError('Expected symmetric nonnegative matrix')
    return TSPInstance(header.get('NAME',path.stem),n,kind,xy,dist)

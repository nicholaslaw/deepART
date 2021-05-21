import numpy as np
import random


class Dataset:

        def __init__(self,data):
                self.data = np.array(data,dtype=np.float32)
                self.data_normalized = np.array(data, dtype=np.float32)
                self.data_normalized[...,0] = self.data[...,0]/max(self.data[...,0])
                self.data_normalized[...,1] = self.data[...,1]/max(self.data[...,1])


class Clusters3d:

        def __init__(self, nclusters,spread=0.1, npoints=50, data_range=[0,600]):
                data = self._cluster3d(nclusters,spread=spread, npoints=npoints, data_range=data_range)
                self.data = np.array(data,dtype=np.float32)
                self.data_normalized = np.array(data, dtype=np.float32)
                self.data_normalized[...,0] = self.data[...,0]/max(self.data[...,0])
                self.data_normalized[...,1] = self.data[...,1]/max(self.data[...,1])
                self.data_normalized[...,2] = self.data[...,2]/max(self.data[...,2])

        
        def _cluster3d(self,nclusters,spread=0.1, npoints=50, data_range=[0,100]):
                '''
                Returns a list of dimension ndata x 3 with coordinates of random points clustered into nclusters
                '''
                # Generate cluster data
                data = []

                for _ in range(nclusters):
                        rx= random.randint(data_range[0],data_range[1])
                        ry= random.randint(data_range[0],data_range[1])
                        rz= random.randint(data_range[0],data_range[1])
                        for _ in range(npoints):
                                deltax = spread*random.randint((data_range[0]-rx), (data_range[1]-rx))
                                deltay = spread*random.randint((data_range[0]-ry), (data_range[1]-ry))
                                deltaz = spread*random.randint((data_range[0]-rz), (data_range[1]-rz))
                                data.append([rx+deltax,ry+deltay,rz+deltaz])
    
                return data

        def addOutlier(self):
                outlierLab = [self.data[0][0]/2, self.data[self.data.shape[0]-1][1]/2, self.data[0][0]/2]
                self.data = np.vstack((self.data, np.array(outlierLab)))
                self.data_normalized = np.array(self.data, dtype=np.float32)
                self.data_normalized[...,0] = self.data[...,0]/max(self.data[...,0])
                self.data_normalized[...,1] = self.data[...,1]/max(self.data[...,1])
                self.data_normalized[...,2] = self.data[...,2]/max(self.data[...,2])

class Clusters2d_overlap:

        def __init__(self, nclusters,overlap=0.8, spread=0.2, npoints=50, data_range=[0,600]):
                self.y = []
                data = self._cluster2d_overlap(nclusters,overlap=overlap,spread=spread, npoints=npoints, data_range=data_range)
                self.data = np.array(data,dtype=np.float32)
                self.data_normalized = np.array(data, dtype=np.float32)
                self.data_normalized[...,0] = self.data[...,0]/max(self.data[...,0])
                self.data_normalized[...,1] = self.data[...,1]/max(self.data[...,1])


        
        def _cluster2d_overlap(self,nclusters,overlap=0.8,spread=0.2, npoints=50, data_range=[0,100]):
                '''
                Returns a list of dimension ndata x 3 with coordinates of random points clustered into nclusters
                '''
                # Generate cluster data
                data = []
                #generate cluster #0
                
                rx= random.randint(data_range[0],data_range[1])
                ry= random.randint(data_range[0],data_range[1])
                
                
                for _ in range(npoints):
                        '''
                        deltax = spread*random.randint((data_range[0]-rx), (data_range[1]-rx))
                        deltay = spread*random.randint((data_range[0]-ry), (data_range[1]-ry))
                        '''
                        data.append([np.random.normal(rx,spread),np.random.normal(ry,spread)])
                        self.y.append(0)
                #generate other clusters
                data_x = [p[0] for p in data]
                data_y = [p[1] for p in data]

                for c in range(nclusters-1):
                        rxp = random.randint(int(min((4-overlap)*(min(data_x)),data_range[0])), int(min((4-overlap)*max(data_x),data_range[1])))
                        ryp = random.randint(int(min((4-overlap)*min(data_y),data_range[0])), int(min((4-overlap)*max(data_y),data_range[1])))
                        
                        for _ in range(npoints):
                                '''
                                deltax = spread*random.randint(int(data_range[0]-rxp), int(data_range[1]-rxp))
                                deltay = spread*random.randint(int(data_range[0]-ryp), int(data_range[1]-ryp))
                                '''
                                data.append([np.random.normal(rxp,spread),np.random.normal(ryp,spread)])
                                self.y.append(c+1)

    
                return data

        def addOutlier(self):
                outlierLab = [self.data[0][0]/2, self.data[self.data.shape[0]-1][1]/2, self.data[0][0]/2]
                self.data = np.vstack((self.data, np.array(outlierLab)))
                self.data_normalized = np.array(self.data, dtype=np.float32)
                self.data_normalized[...,0] = self.data[...,0]/max(self.data[...,0])
                self.data_normalized[...,1] = self.data[...,1]/max(self.data[...,1])
                

class TwoSpirals:

        def __init__(self, n_points, noise=.75):
                X, y = self._twospirals(n_points=n_points,noise=noise)
                self.data = np.array(X,dtype=np.float32)
                self.data_normalized = np.array(X, dtype=np.float32)
                self.data_normalized[...,0] = (self.data[...,0]-min(self.data[...,0]))/2/max(self.data[...,0])
                self.data_normalized[...,1] = (self.data[...,1]-min(self.data[...,1]))/2/max(self.data[...,1])
                self.y = y.astype(int)

        def _twospirals(self,n_points, noise=.75):
                """
                Returns the two spirals dataset.
                """
                n = np.sqrt(np.random.rand(n_points,1)) * 780 * (2*np.pi)/360
                d1x = -np.cos(n)*n + np.random.rand(n_points,1) * noise
                d1y = np.sin(n)*n + np.random.rand(n_points,1) * noise
                return (np.vstack((np.hstack((d1x,d1y)),np.hstack((-d1x,-d1y)))), 
                        np.hstack((np.zeros(n_points),np.ones(n_points))))

        def addOutlier(self):
                outlier = [min(self.data[...,0]),min(self.data[...,1])]
                self.data = np.vstack((self.data, np.array(outlier)))
                self.data_normalized = np.array(self.data, dtype=np.float32)
                self.data_normalized[...,0] = (self.data[...,0]-min(self.data[...,0]))/2/max(self.data[...,0])
                self.data_normalized[...,1] = (self.data[...,1]-min(self.data[...,1]))/2/max(self.data[...,1])
                self.y = np.hstack((self.y,2))
                
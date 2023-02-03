This package provides an implementation of the TopPush algorithm proposed in [1]. 

We provide a demo implementation of TopPush, and develop this package using MATLAB as well
as C-mex (for the projection step). The purpose of this package is to show the effectiveness of 
TopPush, and we believe that better efficiency can be obtained by sophisticated coding tricks.

______________________________________________________________________________________
---- * Manual * ----------------------------------------------------------------------
The main algorithm is given in topPush.m, whose manual is as following: 

Syntax: 
  w = topPush(X, y, opt)

Description:
  topPush takes, 
      X    -    the instance matrix, each row is an instance
      y    -    the labels for each instance (+1 / -1)
      opt  -    the structure of topPush training setting where
          .lambda   - regularization parameter (default: 1)
          .maxIter  - maximum number of iterations (defalut: 10000)
          .tol      - precision parameter (default: 10^-4)
          .debug    - the indictor for debugging (default: false)
                      (true for displaying some inner status)

   and returns, 
      w    - learnt linear ranking model

_________________________________________________________________________________________
----- * DEMO * --------------------------------------------------------------------------
A demo script named 'demo_topPush.m' is provided. It runs topPush on the spambase dataset 

_________________________________________________________________________________________
---- * Projection * ---------------------------------------------------------------------
We implement the projection step (see [1] for details) using C and mex. 
    epne.c          - the C mex-file codes for the projection step
    epne.mexw64     - the complied mex file on Windows (64-bit) 
                      
If you want to run this code on other platforms, please complie 'epne.c' using mex. 

_________________________________________________________________________________________
---- * Attention* -----------------------------------------------------------------------
This package was developed by Mr. Nan Li (lin@lamda.nju.edu.cn). For any problem concerning 
the codes, please feel free to contact Mr. Li.

Reference: 
[1] N. Li, R. Jin and Z.-H. Zhou. Top Rank Optimization in Linear Time. In NIPS-2014.  
    (Long version: CoRR, abs/1410.1462 | http://arxiv.org/abs/1410.1462)

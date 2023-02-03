/* Efficient Projection with Non-negative & Equality Contraint.c
 * Author: Nan LI
 *
 *   min 0.5*|a-a0|^2 + 0.5*|q-q0|^2  st a>=0   ssq>=0   1'a=1'q
 *
 * Input:  a0, q0
 * Output:  a, q
 */

#include <math.h>
#include "mex.h"
#include <string.h>

static void computeProjection(double* a0, double* q0, double* a, double *q, mwIndex m, mwIndex n)
{
    int nga=0, nla=0, nea=0, ngq=0, nlq=0, neq=0, nua=m, nuq=n, i, j, k=0, l=0;
    double *vga=(double*)mxMalloc(sizeof(double)*m);
    double *vla=(double*)mxMalloc(sizeof(double)*m);
    double *vgq=(double*)mxMalloc(sizeof(double)*n);
    double *vlq=(double*)mxMalloc(sizeof(double)*n);
    double *vua=a0, *vuq=q0;
    double sa=0.0, sq=0.0, sq0=0.0, rho, val, dsa, dsq, df;
    
    for(i=0;i<n;i++){// compute the sum of q0
        q0[i]=-q0[i];    sq0+=q0[i]; 
    }
    while(nua+nuq>0){
        if (nua>0) rho=vua[0]; else rho=vuq[0];    //select the threshold
        dsa=0.0; dsq=0.0;
        nga=0; nla=0; nea=0;
        ngq=0; nlq=0; neq=0;
        for(i=0;i<nua;i++){   // split the vector a
            val=vua[i];
            if (val>rho)        {vga[nga]=val; nga++; dsa+=val;} 
            else if (val<rho)   {vla[nla]=val; nla++; }
            else                {dsa+=val; nea++;} 
        }
        for(i=0;i<nuq;i++){   // split the vector q
            val=vuq[i];
            if (val>rho)        {vgq[ngq]=val; ngq++; dsq+=val;} 
            else if (val<rho)   {vlq[nlq]=val; nlq++; }
        }
        df = sa+dsa+(sq0-sq-dsq)-(k+nga+nea)*rho-(n-l-ngq)*rho;        
        if (df<0){
            vua=vla;  nua=nla;  sa+=dsa; k+=(nga+nea);
            vuq=vlq;  nuq=nlq;  sq+=dsq; l+=ngq;
        }else{
            vua=vga;  nua=nga;  
            vuq=vgq;  nuq=ngq; 
        }   
    }
    rho = (sa+sq0-sq)/(k+n-l);
    for(i=0;i<m;i++) {
        val=a0[i]-rho;
        if (val>0) a[i]=val; else a[i]=0.0;
    }
    for(i=0;i<n;i++) {
        q0[i]=-q0[i];
        val=q0[i]+rho;
        if (val>0) q[i]=val; else q[i]=0.0;
    }
    mxFree(vga);mxFree(vla);mxFree(vgq);mxFree(vlq);
}

void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray*prhs[] )
{
    double *p_a0, *p_q0, *p_a, *p_q;
    mwIndex m, n;
    
    m = mxGetM(prhs[0]);
    plhs[0] = mxCreateDoubleMatrix(m, (mwIndex)1, mxREAL);
    if (plhs[0]==NULL) {
        fprintf(stderr, "epne.c: Out of Memory!");
        return;
    }
    
    n = mxGetM(prhs[1]);
    plhs[1] = mxCreateDoubleMatrix(n, (mwIndex)1, mxREAL);
    if (plhs[1]==NULL) {
        fprintf(stderr, "epne.c: Out of Memory!");
        return;
    }
    
    p_a0 = mxGetPr(prhs[0]);
    p_q0 = mxGetPr(prhs[1]);
    p_a  = mxGetPr(plhs[0]);
    p_q  = mxGetPr(plhs[1]);
    computeProjection(p_a0, p_q0, p_a, p_q, m, n);
}

#ifndef _FILTERS_H
#define _FILTERS_H

double laplace3[3] 	= {1,-2,1};
double test[3] 		= {1,0,1};

double* ones(int N)
{
    double *filter = (double*) malloc(N * sizeof(double));

    int i;

    for(i=0;i<N;i++)
    {
        filter[i] = 1;
    }

    return filter; 
}

#endif 
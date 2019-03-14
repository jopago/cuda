#ifndef _FILTERS_2D_H
#define _FILTERS_2D_H

double laplace2d[9] =
{
	0,1,0,
	1,-4,1,
	0,1,0
};

double ones2d[9] =
{
	1,1,1,
	1,1,1,
	1,1,1
};

double fd_x[9] =
{
	0,0,0,
	1,0,-1,
	0,0,0
};

double fd_y[9] =
{
	0,1,0,
	0,0,0,
	0,-1,0
};

#endif
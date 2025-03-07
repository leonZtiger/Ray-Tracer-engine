﻿#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <string>
#include <sstream>
#include<fstream>
#include"window.h"
#include<vector>
#include<algorithm>
#include<iostream>
#include"sprite.h"
#include"memManager.h"
#include<stdlib.h>
#include"curand_kernel.h"

#ifndef max

#define max( a, b ) ( ((a) > (b)) ? (a) : (b) )

#endif

#ifndef min

#define min( a, b ) ( ((a) < (b)) ? (a) : (b) )

#endif
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

using namespace std;


struct vec2d {
float u, v;
};
struct matrix {
	float mat[4][4] = { 0 };
};
struct vec3d {
	float x, y, z;
};
__device__ __host__
static int	getVec3dSize() {
	return sizeof(float) * 3;
}
__device__ __host__
vec3d sub(vec3d& vec1, vec3d& vec2) {

	return { vec1.x - vec2.x, vec1.y - vec2.y, vec1.z - vec2.z };
}
__device__ __host__
vec3d sub(float k, vec3d& vec2) {

	return { k - vec2.x, k - vec2.y, k - vec2.z };
}
__device__ __host__
vec3d divide(vec3d& vec1, float k) {

	return { vec1.x / k, vec1.y / k, vec1.z / k};
}
__device__ __host__
vec3d add(vec3d& vec1, vec3d& vec2) {

	return { vec1.x + vec2.x, vec1.y + vec2.y, vec1.z + vec2.z };
}
__device__
vec3d add(vec3d& vec1, float a) {
	return { vec1.x + a, vec1.y + a, vec1.z + a };
}

__device__ __host__
vec3d multiply(vec3d& vec1, vec3d& vec2) {

	return { vec1.x * vec2.x, vec1.y * vec2.y, vec1.z * vec2.z };
}
__device__ __host__
vec3d  multiply(vec3d vec1, float b) {

	return { vec1.x * b, vec1.y * b, vec1.z * b };
}
__device__ __host__
vec3d cross(vec3d& vec1, vec3d& vec2) {

	return { vec1.y * vec2.z - vec1.z * vec2.y,
		vec1.z * vec2.x - vec1.x * vec2.z,
		vec1.x * vec2.y - vec1.y * vec2.x };
}
__device__
float dotproduct(vec3d& vec1) {

	return (vec1.x * vec1.x + vec1.y * vec1.y + vec1.z * vec1.z);
}
__device__ __host__
float dotproduct(vec3d& vec1, vec3d& vec2) {

	return (vec1.x * vec2.x + vec1.y * vec2.y + vec1.z * vec2.z);
}
__device__ __host__
float length(vec3d& vec) {
	return sqrtf(dotproduct(vec, vec));
}
__device__  __host__
vec3d normalise(vec3d& vec1) {
	double l = length(vec1);
	if (l != 0)
		return { vec1.x /= l, vec1.y /= l , vec1.z /= l };
	else
		return vec3d({ 0,0,0 });
}
__device__ __host__
matrix multiply(matrix mat2, matrix mat1) {
	matrix mat;
	for (int i = 0; i < 4; i++)
		for (int j = 0; j < 4; j++)
			mat.mat[j][i] = mat1.mat[j][0] * mat2.mat[0][i] + mat1.mat[j][1] * mat2.mat[1][i] + mat1.mat[j][2] * mat2.mat[2][i] + mat1.mat[j][3] * mat2.mat[3][i];

	return mat;

}
__device__ __host__
vec3d multiply(matrix mat, vec3d v1) {

	vec3d v2;
	v2.x = v1.x * mat.mat[0][0] + v1.y * mat.mat[1][0] + v1.z * mat.mat[2][0];
	v2.y = v1.x * mat.mat[0][1] + v1.y * mat.mat[1][1] + v1.z * mat.mat[2][1];
	v2.z = v1.x * mat.mat[0][2] + v1.y * mat.mat[1][2] + v1.z * mat.mat[2][2];

	return v2;
}
void setIdentity(matrix& mat) {

	mat.mat[0][1] = 1;
	mat.mat[1][2] = 1;
	mat.mat[2][3] = 1;
	mat.mat[3][4] = 1;

}
matrix pointAt(vec3d pos, vec3d target, vec3d up) {

	vec3d forward = normalise(sub(target, pos));

	vec3d right = cross(normalise(up), forward);
	vec3d newUp = cross(forward, right);

	matrix mat;

	mat.mat[0][0] = right.x;
	mat.mat[0][1] = right.y;
	mat.mat[0][2] = right.z;

	mat.mat[1][0] = newUp.x;
	mat.mat[1][1] = newUp.y;
	mat.mat[1][2] = newUp.z;

	mat.mat[2][0] = forward.x;
	mat.mat[2][1] = forward.y;
	mat.mat[2][2] = forward.z;

	mat.mat[3][0] = pos.x;
	mat.mat[3][1] = pos.y;
	mat.mat[3][2] = pos.z;

	return mat;

}
__device__
matrix rotateY(float deg) {
	matrix mat;

	mat.mat[0][0] = cos(deg);
	mat.mat[2][0] = sin(deg);
	mat.mat[0][2] = sin(deg);
	mat.mat[1][1] = 1.f;
	mat.mat[2][2] = cos(deg);
	mat.mat[3][3] = 1.f;

	return mat;
}
__device__
matrix rotateX(float deg) {
	matrix mat;

	mat.mat[0][0] = 1;
	mat.mat[1][1] = cos(deg);
	mat.mat[1][2] = sin(deg);
	mat.mat[2][1] = -sin(deg);
	mat.mat[2][2] = cos(deg);
	mat.mat[3][3] = 1.f;

	return mat;
}
__device__
matrix rotateZ(float deg) {
	matrix mat;

	mat.mat[0][0] = cos(deg);
	mat.mat[0][1] = sin(deg);
	mat.mat[1][0] = -sin(deg);
	mat.mat[1][1] = cos(deg);
	mat.mat[2][2] = 1;
	mat.mat[3][3] = 1;

	return mat;
}


struct triangle {

	vec3d points[3];
	vec3d normal;
	vec3d vecNormal[3];
	vec2d vt[3];
};
class material : public memManager{

public:
	material(float r, float t,float rf) {
		this->reflectivness = r;
		this->transperancy = t;
		this->roughness = rf;
	}
	float reflectivness;
	float transperancy; 
	float roughness;
};
class ray {
public:
	__device__
		ray() {}
	__device__
		ray(vec3d org, vec3d dir) {
		Org = org;
		Dir = dir;
	}
	vec3d Org;
	vec3d Dir;
};
class camera : public ray {

public:
	camera(vec3d pos, vec3d dir,float apature) {
		
		this->Org = pos;
		this->Dir = dir;
	
	}

	__device__
	vec3d rotateDir(vec3d &vec,float yaw,float pitch) {
		float yawRad = yaw * (3.1415 / 180);
		float pitchRad = pitch * (3.1415 / 180);

		float y = vec.y * cosf(pitchRad) - vec.z * sinf(pitchRad);
		float z = vec.y * sinf(pitchRad) + vec.z * cosf(pitchRad);
		float x = vec.x * cosf(yawRad) + z * sinf(yawRad);
		z = -vec.x * sinf(yawRad) + z * cosf(yawRad);

		return { x,y,z };
	}

	float aspect;
	float Camyaw = 180, Campitch = -20;
};


class shape : public memManager
{

public:

	    virtual int getBytes() { return 0; }
	
//	    __device__
//		virtual bool intersect(ray& cam_ray, float& t) { return true; };

	vec3d orgin;

protected:

};

class sphere : public shape {

public:

	sphere(vec3d org, float r) {
		this->orgin = org;
		this->radius = r * r;
	}

	sphere() {}

	__device__
	bool intersect(ray &cam_ray, float& t) {

	/*	
	// geometric solution 
	float t0, t1;

		vec3d L = sub(orgin,cam_ray.Org);
		float tca = dotproduct(L, cam_ray.Dir);
		if (tca < 0)
			return false;
		float d2 = dotproduct(L, L) - tca * tca;
		if (d2 > radius)
			return false;
		float thc = sqrt(radius - d2);
		t0 = tca - thc;
		t1 = tca + thc;

		if (t0 > t1) {
		float temp = 0;
		t0 = t1;
	    t1 = temp;
		}
	


		if (t0 < 0) {
			t0 = t1;
				return false;
		}
		t = t0;
		return true;
    */
		// quadratic solution
		/*
		(xd^2 + yd^2 + zd^2)* t^2 +
         [2[xd * (xo - a) + yd * (yo - b) + zd *(zo - c)]] * t   +
         [(xo - a)^2 + (yo - b)^2 + (zo - c^)2 - r^2]      * 1
         = 0
		*/
		float A = (cam_ray.Dir.x * (cam_ray.Dir.x) + cam_ray.Dir.y * (cam_ray.Dir.y) + cam_ray.Dir.z * (cam_ray.Dir.z));
		float B = 2 * (cam_ray.Dir.x * (cam_ray.Org.x - orgin.x) + cam_ray.Dir.y * (cam_ray.Org.y - orgin.y) + cam_ray.Dir.z * (cam_ray.Org.z - orgin.z));
		float C = (cam_ray.Org.x - orgin.x)* (cam_ray.Org.x - orgin.x) + (cam_ray.Org.y - orgin.y) * (cam_ray.Org.y - orgin.y) + (cam_ray.Org.z - orgin.z) * (cam_ray.Org.z - orgin.z) - radius*radius;

		t = (-B + sqrt(B * B - 4 * A * C)) / (2 * A);

		if (t == 0.f) {
			return true;
		}

		if (t >= 0.0001) {

			float t2 = (-B - sqrt(B * B - 4 * A * C)) / (2 * A);
			
			if (t > t2) {
				t = t2;

				return true;
			}
			return true;
		}
		return false;
	}
	bool reflective = false;
	float radius;

};

class plane : public shape {

public:

	plane(vec3d pos, vec3d normal) {
		this->orgin = pos;
		this->normal = normal;
	}

	__device__
		bool intersect(ray & cam_ray, float& t) {

		float denom = dotproduct(normal, cam_ray.Dir);

		if (denom < 0) {
			vec3d pl0 = sub(orgin,cam_ray.Org);
			t = dotproduct(pl0, normal)/denom;
			return t >= 0;
		}
		return false;
	}

	bool reflective = false;
	vec3d normal;
};


class cube : public shape{

public:

	cube(vec3d c1,vec3d c2) {

		this->bounds[0] = c1;
		this->bounds[1] = c2;
		this->orgin = divide(add(c1, c2),2);
	}
	cube() {  }
	
	__device__
	bool intersect(ray& cam_ray, float& t) {

		/*vec3d new_dir({(1.f / cam_ray.Dir.x),(1.f / cam_ray.Dir.y),(1.f / cam_ray.Dir.z)});

		int sign[3] = { (new_dir.x < 0),(new_dir.y < 0) ,(new_dir.z < 0) };

		float tmin = (bounds[sign[0]].x - cam_ray.Org.x) * new_dir.x;
		float tmax = (bounds[1 - sign[0]].x - cam_ray.Org.x) * new_dir.x;
		float tymin = (bounds[sign[1]].y - cam_ray.Org.y) * new_dir.y;
		float tymax = (bounds[1 - sign[1]].y - cam_ray.Org.y) * new_dir.y;

		if ((tmin > tymax) || (tymin > tmax))
			return false;

		if (tymin > tmin)
			tmin = tymin;

		if (tymax < tmax)
			tmax = tymax;

		float tzmin = (bounds[sign[2]].z - cam_ray.Org.z) * new_dir.z;
		float tzmax = (1 - bounds[sign[2]].z - cam_ray.Org.z) * new_dir.z;

		if ((tmin > tzmax) || (tzmin > tmax))
			return false;

		if (tzmin > tmin)
			tmin = tzmin;

	//	if (tzmax < tmax)
	//		tmax = tzmax;

			t = tmin;
	
		

		float xmax, xmin,ymax,ymin,zmax,zmin;

		check_axis(cam_ray.Org.x, cam_ray.Dir.x,xmin,xmax);
		check_axis(cam_ray.Org.y, cam_ray.Dir.y, ymin,ymax);
		check_axis(cam_ray.Org.z, cam_ray.Dir.z, zmin, zmax);

		float tmin = xmin > ymin && xmin > zmin? xmin: ymin > xmin && ymin > zmin ? ymin: zmin;
		float tmax = xmax < ymax && xmax < zmax ? xmax : ymax < xmax && ymax < zmax ? ymax : zmax;

		if (tmin >= tmax)
			return false;

	//		float temp = tmax;
	//		tmax = tmin;
	//		tmin = temp;
	
		t = tmin;

		return true;
*/
		float dirx = 1.f / cam_ray.Dir.x;
		float diry = 1.f / cam_ray.Dir.y;
		float dirz = 1.f / cam_ray.Dir.z;

		float t1 = (bounds[0].x - cam_ray.Org.x) * dirx;
		float t2 = (bounds[1].x - cam_ray.Org.x) * dirx;

		float t3 = (bounds[0].y - cam_ray.Org.y) * diry;
		float t4 = (bounds[1].y - cam_ray.Org.y) * diry;

		float t5 = (bounds[0].z - cam_ray.Org.z) * dirz;
		float t6 = (bounds[1].z - cam_ray.Org.z) * dirz;

		float tmin = max(max(min(t1, t2), min(t3, t4)),min(t5,t6));
		float tmax = min(min(max(t1, t2), max(t3, t4)), max(t5, t6));

		if (tmax < 0) {
			t = tmax;
			return false;
		}

		if (tmax < tmin) {
			t = tmax;
			return false;
		}

		t = tmin;
		return true;
	}
	__device__
	void check_axis(float orgin, float dir, float& tmin, float &tmax) {

		float tminN = -1 - orgin;
		float tmaxN = 1 - orgin;


		if (fabs(dir) >= 0.0001) {
			tmin = tminN / dir;
			tmax = tmaxN / dir;
		if (tmin > tmax)
		{
			float temp = tmax;
			tmax = tmin;
			tmin = temp;
		}
		}
		else {
			tmin = INFINITY;
			tmax = INFINITY;
		}
	}
	vec3d normals[3] = {{1,0,0},{0,1,0},{0,0,1}};
	vec3d bounds[2];
};


class Bvhbox : memManager{
public:

	Bvhbox(vec3d c1,vec3d c2,int *in,int len) {

		this->length = len;
		this->indexes = in;
		this->bvhbox = new cube(c1,c2);
	//	cubeAllocMem(new cube( c1,c2),in);
	}
	Bvhbox(){}

	int getBytes() {
		return sizeof(cube);
	}

	void AllocMem() {
		int size = getBytes();
		checkCudaErrors(cudaMallocManaged((void**)&d_bvhbox, size));
		memcpy(d_bvhbox, bvhbox, size);

		checkCudaErrors(cudaMallocManaged((void**)&d_indexes, sizeof(int)*length));
		memcpy(d_indexes, indexes, sizeof(int) * length);
	}


	cube *bvhbox;
	cube *d_bvhbox;
	int *indexes;
	int *d_indexes;
	int length;
};


__device__ __host__
unsigned long rgbToInt(int r, int g, int b) {
	if (r > 255)
		r = 255;
	if (g > 255)
		g = 255;
	if (b > 255)
		b = 255;

	return((r & 0xff) << 16) + ((g & 0xff) << 8) + (b & 0xff);
}


class mesh : public memManager {

public:
    
	
	
	triangle *d_tri_arr;
	triangle* h_tri_arr;

    int poly_count;
	int bvhbox_count;
	int bvhLayer_count = 10;
	bool has_normals = false;

	Bvhbox *h_box;
	Bvhbox *d_box;
    int* indexes;
	
	mesh(string filename)
	{
		
		vector<triangle> tris;
		ifstream file(filename);

		if (!file.is_open()) {
			return;
		}

		vector<vec3d> p;
		vector<vec3d> vn;
		vector<vec2d> vt;

		string line;
		char c;
       
		while (getline(file, line)) {
      
			char junk;
			char type = line[1];
			istringstream  ss(line);
		
			ss >> c;
			
			//if its a vertices value 
			if (c == 'v' || c == 'V') {
			
				if (type == 'n' || type == 'N') {
					
					vec3d n;
					
					ss>>junk >> n.x >> n.y >> n.z;
					
				//	cout << n.x <<" " << n.y << " " << n.z << "\n";
					
						vn.push_back(n);
				}
				else if (type == 't' || type == 'T') {
					vec2d v;

					ss >>junk >> v.u >> v.v;
					vt.push_back(v);
				}
				else {
					vec3d v;
                    
					ss >> v.x >> v.y >> v.z;
					p.push_back(v);
				}
			}

		if (c == 'f' || c == 'F') {


				string in[4];

				ss >> in[0] >> in[1] >> in[2] >> in[3];
		//		cout << in[0]<<"  " << in[1] << "  " << in[2] << "\n";
                if (vn.size() != 0 && vt.size() != 0) {
				
					if (SlashCount(line) <= 6) {
               
					
					int f[3];  
					int pvt[3]; 
					int pvn[3]; 
         			
						

					splitString(in[0]) >> f[0] >> pvt[0] >> pvn[0];
					splitString(in[1]) >> f[1] >> pvt[1] >> pvn[1];
					splitString(in[2]) >> f[2] >> pvt[2] >> pvn[2];
			 
					
						
					vec3d normal = cross(sub(p[f[1] - 1], p[f[0] - 1]), sub(p[f[2] - 1], p[f[0] - 1]));
					normal = normalise(normal);

					tris.push_back({ p[f[0] - 1],p[f[1] - 1] ,p[f[2] - 1] , normal,vn[pvn[0] - 1],vn[pvn[1] - 1],vn[pvn[2] - 1] ,vt[pvt[0]-1],vt[pvt[1] - 1] ,vt[pvt[2] - 1] });

				}
				if (SlashCount(line) >= 8) {

					int f[4];
					int pvt[4];
					int pvn[4];



					splitString(in[0]) >> f[0] >> pvt[0] >> pvn[0];
					splitString(in[1]) >> f[1] >> pvt[1] >> pvn[1];
					splitString(in[2]) >> f[2] >> pvt[2] >> pvn[2];
					splitString(in[3]) >> f[3] >> pvt[3] >> pvn[3];



					vec3d normal = cross(sub(p[f[1] - 1], p[f[0] - 1]), sub(p[f[2] - 1], p[f[0] - 1]));
					normal = normalise(normal);

					tris.push_back({ p[f[0] - 1],p[f[1] - 1] ,p[f[2] - 1] , normal,vn[pvn[0] - 1],vn[pvn[1] - 1],vn[pvn[2] - 1] ,vt[pvt[0] - 1],vt[pvt[1] - 1] ,vt[pvt[2] - 1] });
					tris.push_back({ p[f[0] - 1],p[f[2] - 1] ,p[f[3] - 1] , normal,vn[pvn[0] - 1],vn[pvn[2] - 1],vn[pvn[3] - 1] ,vt[pvt[0] - 1],vt[pvt[2] - 1] ,vt[pvt[2] - 1] });

					

				}
				has_normals = true;
				}else if (vn.size() != 0) {

					if (SlashCount(line) <= 6) {
                     int f[3];
					int pvn[3];
           
 	       //     	cout << in[0] << "  " << in[1] << "  " << in[2] << "\n";
		


					splitString(in[0]) >> f[0] >> pvn[0];
					splitString(in[1]) >> f[1] >> pvn[1];
					splitString(in[2]) >> f[2] >> pvn[2];

			//		cout << vn[pvn[0]-1].x << "  " << vn[pvn[0] - 1].y << "  " << vn[pvn[0] - 1].z << "\n";

			//		cout << pvn[0] << "  " << pvn[1] << "  " << pvn[2] << "\n";

					vec3d normal = cross(sub(p[f[1] - 1], p[f[0] - 1]), sub(p[f[2] - 1], p[f[0] - 1]));
					normal = normalise(normal);

					tris.push_back({ p[f[0] - 1],p[f[1] - 1] ,p[f[2] - 1] , normal,vn[pvn[0] - 1],vn[pvn[1] - 1],vn[pvn[2] - 1],vec2d({0,0}),vec2d({0,1}),vec2d({1,0}) });
					
					}
					if (SlashCount(line) >= 8) {
						int f[4];
						int pvn[4];

						//     	cout << in[0] << "  " << in[1] << "  " << in[2] << "\n";



						splitString(in[0]) >> f[0] >> pvn[0];
						splitString(in[1]) >> f[1] >> pvn[1];
						splitString(in[2]) >> f[2] >> pvn[2];
						splitString(in[3]) >> f[3] >> pvn[3];

						//		cout << vn[pvn[0]-1].x << "  " << vn[pvn[0] - 1].y << "  " << vn[pvn[0] - 1].z << "\n";

						//		cout << pvn[0] << "  " << pvn[1] << "  " << pvn[2] << "\n";

						vec3d normal = cross(sub(p[f[1] - 1], p[f[0] - 1]), sub(p[f[2] - 1], p[f[0] - 1]));
						normal = normalise(normal);

						tris.push_back({ p[f[0] - 1],p[f[1] - 1] ,p[f[2] - 1] , normal,vn[pvn[0] - 1],vn[pvn[1] - 1],vn[pvn[2] - 1],vec2d({0,0}),vec2d({0,1}),vec2d({1,0}) });
						tris.push_back({ p[f[0] - 1],p[f[2] - 1] ,p[f[3] - 1] , normal,vn[pvn[0] - 1],vn[pvn[1] - 1],vn[pvn[2] - 1],vec2d({0,0}),vec2d({0,1}),vec2d({1,0}) });
						
					 }
					has_normals = true;
				}
				else {
					

					if (SlashCount(line) <= 6) {
                    int f[3] = { stoi(in[0]),stoi(in[1]),stoi(in[2]) };

					vec3d normal = cross(sub(p[f[1] - 1], p[f[0] - 1]), sub(p[f[2] - 1], p[f[0] - 1]));

					normal = normalise(normal);

					tris.push_back({ p[f[0] - 1],p[f[1] - 1] ,p[f[2] - 1] , normal,vec3d({0,0,0}),vec3d({0,0,0}),vec3d({0,0,0}),vec2d({ 0.666413, 0.250594}),vec2d({0.333587 ,0.250594}),vec2d({0.333587,0.000975}) });
					}
					if (SlashCount(line) == 8) {
						int f[4] = { stoi(in[0]),stoi(in[1]),stoi(in[2]),stoi(in[3]) };

						vec3d normal = cross(sub(p[f[1] - 1], p[f[0] - 1]), sub(p[f[2] - 1], p[f[0] - 1]));

						normal = normalise(normal);

						tris.push_back({ p[f[0] - 1],p[f[1] - 1] ,p[f[2] - 1] , normal,vec3d({0,0,0}),vec3d({0,0,0}),vec3d({0,0,0}),vec2d({0,0}),vec2d({0,1}),vec2d({1,0}) });
						tris.push_back({ p[f[0] - 1],p[f[2] - 1] ,p[f[3] - 1] , normal,vec3d({0,0,0}),vec3d({0,0,0}),vec3d({0,0,0}),vec2d({0,0}),vec2d({0,1}),vec2d({1,0}) });

					}
					has_normals = false;
					
			}
			}
		}
		file.close();
		poly_count = tris.size();


		h_tri_arr = new triangle[poly_count];
		indexes = new int[poly_count];

		for (int i = 0; i < poly_count; i++) {

			h_tri_arr[i] = tris[i];
			indexes[i] = i;
		}
		
		
		
		createBvhMesh();

		tris.clear();
	}

	void createBvhMesh() {

		vector<Bvhbox> tempBoxes;
		vector<Bvhbox> prevBoxes;
		int splitDir = 0;


		vec3d lowestP = h_tri_arr[0].points[0];
		vec3d highestP = h_tri_arr[poly_count - 1].points[0];

		//find lowest and highest
		for (int i = 0; i < poly_count; i++) {

			getMinMaxP(h_tri_arr[i], lowestP, highestP);
		}

		prevBoxes.push_back(Bvhbox(lowestP, highestP, indexes, poly_count));

		//	cout << highestP.x << highestP.y << highestP.z<<"\r\n";
		//	cout << lowestP.x << lowestP.y << lowestP.z;


		for (int i = 0; i < bvhLayer_count; i++) {

			for (int j = 0; j < prevBoxes.size(); j++) {

				if (prevBoxes[j].length > 5) {
					vector<int> newIndexes1;
					vector<int> newIndexes2;

					if (splitDir == 0) {

						float splitP = (prevBoxes[j].bvhbox->bounds[1].y + prevBoxes[j].bvhbox->bounds[0].y) / 2;



						for (int z = 0; z < prevBoxes[j].length; z++) {

							if (h_tri_arr[prevBoxes[j].indexes[z]].points[0].y <= splitP) {
								newIndexes1.push_back(prevBoxes[j].indexes[z]);
							}
							else {
								newIndexes2.push_back(prevBoxes[j].indexes[z]);
							}
						}
					}
					if (splitDir == 1) {

						float splitP = (prevBoxes[j].bvhbox->bounds[1].x + prevBoxes[j].bvhbox->bounds[0].x) / 2;



						for (int z = 0; z < prevBoxes[j].length; z++) {

							//	for (int p = 0; p < p; i++) {


							if (h_tri_arr[prevBoxes[j].indexes[z]].points[0].x <= splitP) {
								newIndexes1.push_back(prevBoxes[j].indexes[z]);
								//	  break;
							}
							else {
								newIndexes2.push_back(prevBoxes[j].indexes[z]);

							}

							//		  break;
						//		}
						}
						//    cout << prevBoxes[j].length << "\r\n";
					}
					if (splitDir == 2) {

						float splitP = (prevBoxes[j].bvhbox->bounds[1].z + prevBoxes[j].bvhbox->bounds[0].z) / 2;



						for (int z = 0; z < prevBoxes[j].length; z++) {

							//	for (int p = 0; p < p; i++) {


							if (h_tri_arr[prevBoxes[j].indexes[z]].points[0].z <= splitP) {
								newIndexes1.push_back(prevBoxes[j].indexes[z]);
								//	  break;
							}
							else {
								newIndexes2.push_back(prevBoxes[j].indexes[z]);

							}

							//		  break;
						//		}
						}
						//    cout << prevBoxes[j].length << "\r\n";
					}

					if (newIndexes1.size() != 0) {

						vec3d low = h_tri_arr[newIndexes1[0]].points[0];
						vec3d high = h_tri_arr[newIndexes1[newIndexes1.size() - 1]].points[0];

						int* toPointerArray = new int[newIndexes1.size()];

						//get each bound of the new bvhboxes
						for (int z = 0; z < newIndexes1.size(); z++) {

							getMinMaxP(h_tri_arr[newIndexes1[z]], low, high);
							toPointerArray[z] = newIndexes1[z];
						}

						tempBoxes.push_back(Bvhbox(low, high, toPointerArray, newIndexes1.size()));
					}
					if (newIndexes2.size() != 0) {
						vec3d low = h_tri_arr[newIndexes2[0]].points[0];
						vec3d high = h_tri_arr[newIndexes2[newIndexes2.size() - 1]].points[0];

						int* toPointerArray = new int[newIndexes2.size()];
						//get each bound of the new bvhboxes
						for (int z = 0; z < newIndexes2.size(); z++) {

							getMinMaxP(h_tri_arr[newIndexes2[z]], low, high);
							toPointerArray[z] = newIndexes2[z];
						}

						tempBoxes.push_back(Bvhbox(low, high, toPointerArray, newIndexes2.size()));
					}
					splitDir = (splitDir == 0 ? 1 : splitDir == 1 ? 2 : 0);
				}
                else {
				tempBoxes.push_back(prevBoxes[j]);
                }
			}
				prevBoxes.clear();
				prevBoxes = tempBoxes;
				tempBoxes.clear();
		}
			//split
			// 0 = vertical 0deg , 1 = horizontal, 2 = vertical 90deg  

		//	h_box[0] = Bvhbox(lowestP, highestP, indexes, poly_count);
			bvhbox_count = prevBoxes.size();
			h_box = new Bvhbox[bvhbox_count];

			for (int j = 0; j < prevBoxes.size(); j++) {

				cout << prevBoxes[j].bvhbox->bounds[1].x <<"\r\n";
				cout << prevBoxes[j].bvhbox->bounds[1].y << "\r\n";
				cout << prevBoxes[j].bvhbox->bounds[1].z << "\r\n";

				h_box[j] = prevBoxes[j];
				h_box[j].AllocMem();
			}
			cout << bvhbox_count;
		}
	bool isgreater(vec3d vec1, vec3d vec2) {

		float a[3]{vec1.x, vec1.y, vec1.z}; 
		float b[3]{ vec2.x, vec2.y, vec2.z };

		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				if (a[i] > b[j])
					return true;
			}
		}
		return false;
	}
	
	vec3d getBiggest(triangle tri) {
	
		if (isgreater(tri.points[0], tri.points[1]) && isgreater(tri.points[0], tri.points[2])) {
			return tri.points[0];
		}
		if (isgreater(tri.points[1], tri.points[0]) && isgreater(tri.points[1], tri.points[2])) {
			return tri.points[1];
		}
		return tri.points[2];
	}

	vec3d getLowest(triangle tri) {

		if (isgreater(tri.points[2], tri.points[0]) && isgreater(tri.points[1], tri.points[0])) {
			return tri.points[0];
		}
		if (isgreater(tri.points[0], tri.points[1]) && isgreater(tri.points[2], tri.points[0])) {
			return tri.points[1];
		}
		return tri.points[2];
	}

	void getMinMaxP(triangle tri, vec3d &min,vec3d &max) {

		for (int j = 0; j < 3; j++) {

			if (tri.points[j].x > max.x) {
				max.x = tri.points[j].x;
			}
			if (tri.points[j].y > max.y) {
				max.y = tri.points[j].y;
			}
			if (tri.points[j].z > max.z) {
				max.z = tri.points[j].z;
			}

			if (tri.points[j].x < min.x) {
				min.x = tri.points[j].x;
			}
			if (tri.points[j].y < min.y) {
				min.y = tri.points[j].y;
			}
			if (tri.points[j].z < min.z) {
				min.z = tri.points[j].z;
			}
		}
	}

	void allocMem() {
		
	/*	// move bvh array to device
		int Bsize = sizeof(float)*6*bvhbox_count+sizeof(int)+sizeof(int)*1;
		checkCudaErrors(cudaMallocManaged((void**)&d_box, Bsize));
		memcpy(d_box, h_box, Bsize);*/

		// move triangle array to device
		int size = getBytes();
		checkCudaErrors(cudaMallocManaged((void**)&d_tri_arr, size));
		memcpy(d_tri_arr, h_tri_arr, size);

		int Bsize = sizeof(Bvhbox)*bvhbox_count;
		checkCudaErrors(cudaMallocManaged((void**)&d_box, Bsize));
		//memcpy(d_box, h_box, Bsize);
		for(int i = 0; i < bvhbox_count; i++) {
			d_box[i] = h_box[i];
		}
	}

	 __host__
	int getBytes() {
		return ((sizeof(float) * 3*7)+(sizeof(float)*2*3)) * poly_count;
	}	

	 __device__
	 bool rayIntersect(ray& ray, triangle& tri, float& t, float& u, float& v) {

		 vec3d edge1 = sub(tri.points[1], tri.points[0]);
		 vec3d edge2 = sub(tri.points[2], tri.points[0]);
		 vec3d h, s, q;
		 float a, f;

		 h = cross(ray.Dir, edge2);
		 a = dotproduct(edge1, h);

		 if (a > -0.0000001f && a < 0.0000001)
			 return false;

		 f = 1.f / a;

		 s = sub(ray.Org, tri.points[0]);
		 u = f * dotproduct(s, h);

		 if (u < 0.f || u> 1.f)
			 return false;

		 q = cross(s, edge1);
		 v = f * dotproduct(ray.Dir, q);
		 if (v < 0.f || u + v>1.f)
			 return false;

		 t = f * dotproduct(edge2, q);
		 if (t > 0.0000001) {
			 return true;
		 }
		 else {
			 return false;
		 }

	 }

private:
	
	int SlashCount(string s) {
		int slash_count = 0;
		for (int i = 0; i < s.length(); i++) {
			if (s[i] == '/')
				slash_count++;
		}
		return slash_count;
	}

	stringstream splitString(string s) {

		stringstream temp;

		int i = 0;
		int v_count = 0;

		while ((s.length() ) > i) {

			if (s[i] != '/') {
				v_count++;
				temp << s[i];
			}
			else {
				temp << " ";
			}
			i++;

		}

	//	int val[3];

	//	if (v_count == 2) {
	//	cout << temp;

	//	temp >> val[0] >> val[1] >> val[2];
	//	cout << val[0] << "  " << val[1] << " "<< val[2]<< "\n";

		return temp;
/*}
		if (v_count == 1) {

			temp >> val[0] >> val[1];
			
			return val;
		}*/ 
			
 }
protected :

	
};


class skybox : public memManager{

public:

	skybox(string img, float size) {
		skyboxTex = new sprite(img);
		box = new sphere({0,0,0},size);
	}

	
	__device__
	void getColor(ray &in_ray,int &r, int &g, int &b) {
		
		float t;

		box->intersect(in_ray, t);
			vec3d hit_point = add(in_ray.Org, multiply(in_ray.Dir, t));
			vec3d normal = sub(hit_point, box->orgin);
			normal = normalise(normal);

			int x = (int)((1 + atan2(normal.z, normal.x) / 3.1415) * 0.5 *skyboxTex->width);
			int y =(int) (acosf(normal.y) / 3.1415 * skyboxTex->height);

			int index = y * skyboxTex->width + x;

			r = skyboxTex->rBuff->data[index];
			g = skyboxTex->gBuff->data[index];
			b = skyboxTex->bBuff->data[index];
		
	}
	__device__
		void getFColor(ray& in_ray, float& r, float& g, float& b) {

		float t;

		box->intersect(in_ray, t);

			vec3d hit_point = add(in_ray.Org, multiply(in_ray.Dir, t));
			vec3d normal = sub(hit_point, box->orgin);
			normal = normalise(normal);

			int x = ((1.f + atan2(normal.z, normal.x) / 3.1415f) * 0.5f * skyboxTex->width);
			int y = (acosf(normal.y) / 3.1415f * skyboxTex->height);

			int index = y * skyboxTex->width + x;

			r = skyboxTex->rBuff->data[index];
			g = skyboxTex->gBuff->data[index];
			b = skyboxTex->bBuff->data[index];
		
	}
sphere* box;
protected:
	
	sprite* skyboxTex;
	

};


class object : public memManager {

public:
	object() {}

	void loadMesh(string file, string tex, material mat) {

		mesh1 = new mesh(file);
		mesh1->allocMem();
		s1 = new sphere[sphere_count];
		c1 = new cube[cube_count];
		planes = new plane({ 0,-4,0 }, normalise(vec3d({ 0,1, 0 })));

		for (int i = 0; i < sphere_count; i++) {
			s1[i] = sphere({ (float)(rand() % 100) / 10 ,(float)(rand() % 100) / 10,(float)(rand() % 100) / 10 }, (float)(rand() % 100) / 100);
				
		}
		for (int i = 0; i < cube_count; i++) {
			float posx = (float)(rand() % 100) / 0.9;
			float posy = (float)(rand() % 100) / 0.9;
			float posz = (float)(rand() % 100) / 0.9;
			c1[i] = cube({ posx,posy,posz }, { posx-2,posy-2,posz-2 });
			//c1[i] = cube({ 0,0,0 }, { 5,5,5 });
		}
		
		texture = new sprite(tex);
		this->mat = new material(mat);
		sphereAllocMem();
		planeAllocMem();	
		cubeAllocMem();

	}
	void sphereAllocMem() {
		int size = Sphere_getBytes();
		checkCudaErrors(cudaMallocManaged((void**)&d_spheres, size));
		memcpy(d_spheres, s1, size);
	}
	void planeAllocMem() {
		int size = sizeof(float) * 10;
		checkCudaErrors(cudaMallocManaged((void**)&d_planes, size));
		memcpy(d_planes, planes, size);
	}
	int Sphere_getBytes() {
		return sizeof(float) * 8 * sphere_count;
	}
	int Cube_getBytes() {
		return sizeof(cube) * cube_count;
	}
	void cubeAllocMem() {
		int size = Cube_getBytes();
		checkCudaErrors(cudaMallocManaged((void**)&d_cubes, size));
		memcpy(d_cubes, c1, size);
	}
	

	int sphere_count = 0, plane_count = 0,cube_count = 0;
	int depth = 3;
	sphere* s1;
	sphere* d_spheres;
	cube* c1;
	cube* d_cubes;
	plane* planes;
	plane* d_planes;
	mesh* mesh1;
	sprite* texture;
	material* mat;
	mesh** tot_mesh;
	int meshes;
};

class light {

public :

	light(){}
	light(vec3d pos,float size,float r, float g, float b) {
		this->pos = pos;
		this->size = size;
		this->r = r;
		this->g = g;
		this->b = b;
	}

	vec3d pos;
	float size,r,g,b;
};
__device__ __host__
matrix rotate(float angle, vec3d vec) {

	matrix rot;

	rot.mat[0][0] = cosf(angle) + vec.x * vec.x;
	rot.mat[0][1] = vec.x * vec.y*(1.f-cosf(angle))-vec.z*sinf(angle);
	rot.mat[0][2] = vec.x * vec.z * (1.f - cosf(angle)) - vec.y * sinf(angle);

	rot.mat[1][0] = vec.y * vec.x * (1.f - cosf(angle)) + vec.z * sinf(angle);
	rot.mat[1][1] = cosf(angle)+ vec.y*vec.y * (1.f - cos(angle));
	rot.mat[1][2] = vec.y * vec.z * (1.f - cosf(angle)) - vec.x * sinf(angle);

	rot.mat[2][0] = vec.z * vec.x * (1.f - cosf(angle)) - vec.y * sinf(angle);
	rot.mat[2][1] = vec.z * vec.y *(1.f -cosf(angle)) + vec.x*sinf(angle);
	rot.mat[2][2] = cosf(angle) + vec.z*vec.z*(1.f -cosf(angle));

	return rot;
}

__device__
vec3d reflect(vec3d &I, vec3d &N) {
	return sub(I,multiply(multiply(N,dotproduct(I,N)),2));
}

__device__
bool castRay(object& objs, ray& cam_ray, int& hit_type, int& hit_index, float& nt, float& nu, float& nv, vec3d& new_org, vec3d& normal, float& tx, float& ty) {

	nt = INFINITY;
	//check for triangle intersection
	float bvht = INFINITY;
	for (int j = 0; j < objs.mesh1->bvhbox_count; j++)
	{
		float temp;

		if (objs.mesh1->d_box[j].d_bvhbox->intersect(cam_ray, temp)) {
			
			//if (temp < bvht) {
				
			//	bvht = temp;

				for (int i = 0; i < objs.mesh1->d_box[j].length; i++)
				{
					float t, u, v;

					if (objs.mesh1->rayIntersect(cam_ray, objs.mesh1->d_tri_arr[objs.mesh1->d_box[j].d_indexes[i]], t, u, v))
					{
						if (t < nt)
						{
							nt = t;
							nv = v;
							nu = u;
							hit_index = objs.mesh1->d_box[j].d_indexes[i];//objs.mesh1->d_box[j].d_indexes[i];
							hit_type = 0;
						}
		//			}
				}
			}
		/*	if (temp < nt)
			{
				nt = temp;
				
				hit_index = objs.mesh1->d_box[j].d_indexes[0];//objs.mesh1->d_box[j].d_indexes[i];
				hit_type = 0;
			}*/
	  }
	}
		//checkfor sphere intersection
		for (int i = 0; i < objs.sphere_count; i++)
		{
			float t;
			if (objs.d_spheres[i].intersect(cam_ray, t))
			{
				if (t < nt)
				{
					nt = t;
					hit_index = i;
					hit_type = 1;
				}
			}
		}
		//check for cube intersection
		for (int i = 0; i < objs.cube_count; i++)
		{
			float t;
			if (objs.d_cubes[i].intersect(cam_ray, t))
			{
				if (t < nt)
				{
					nt = t;
					hit_index = i;
					hit_type = 3;
				}
			}
		}

		//check for plane intersection
		for (int i = 0; i < objs.plane_count; i++)
		{
			float t;

			if (objs.d_planes[i].intersect(cam_ray, t))
			{
				if (t < nt)
				{
					nt = t;
					hit_index = i;
					hit_type = 2;
				}
			}
		}
	
	if (nt != INFINITY) {
		//check what type of object the ray hitted and calculate the normal 
        // 0 = triangle, 1 = sphere, 2 = plane
		//triangle hit
		if (hit_type == 0) {

			if (objs.mesh1->has_normals) {
				normal = add(add(multiply(objs.mesh1->d_tri_arr[hit_index].vecNormal[0], (1 - nu - nv)), multiply(objs.mesh1->d_tri_arr[hit_index].vecNormal[1], nu)), multiply(objs.mesh1->d_tri_arr[hit_index].vecNormal[2], nv));
				normal = normalise(normal);
			}

			else {
				normal = objs.mesh1->d_tri_arr[hit_index].normal;
			}
			//calc the texture vertices
		    tx = ((1 - nu - nv) * objs.mesh1->d_tri_arr[hit_index].vt[0].u) + (nu * objs.mesh1->d_tri_arr[hit_index].vt[1].u) + (nv * objs.mesh1->d_tri_arr[hit_index].vt[2].u);  //(int)(((nu + 1.0f) / 2.0f) * maxX)-1;
			ty = ((1 - nu - nv) * objs.mesh1->d_tri_arr[hit_index].vt[0].v) + (nu * objs.mesh1->d_tri_arr[hit_index].vt[1].v) + (nv * objs.mesh1->d_tri_arr[hit_index].vt[2].v);//((float)maxY -((nv + 1.0f) / 2.0f) * maxY)-1;

			new_org = add(normal, add(cam_ray.Org, multiply(cam_ray.Dir, nt)));
		}

		//sphere hit
		if (hit_type == 1) {

			new_org = add(cam_ray.Org, multiply(cam_ray.Dir, nt));
			normal = sub(new_org, objs.d_spheres[hit_index].orgin);
			normal = normalise(normal);
			//calc the texture vertices
			tx = (1 + atan2(normal.z, normal.x) / 3.1415) * 0.5;
			ty = acosf(normal.y) / 3.1415;

		}
		//plane hit
		if (hit_type == 2) {
			new_org = add(cam_ray.Org, multiply(cam_ray.Dir, nt));
			normal = objs.d_planes[hit_index].normal;
		//	normal = normalise(normal);

			//texture vertices
			tx = 0.5; //hit_point.x;
			ty = 0.5;//hit_point.z* mod(1.f);

		}
		//cube hit
		if (hit_type == 3) {
			new_org = add(cam_ray.Org, multiply(cam_ray.Dir, nt));
			normal = sub(new_org, objs.d_cubes[hit_index].orgin);
			normal = normalise(normal);
			//calc the texture vertices
			tx = (1 + atan2(normal.z, normal.x) / 3.1415) * 0.5;
			ty = acosf(normal.y) / 3.1415;
		}
		
		return true;
	}
	
	return false;
}
__device__
float castLightRay(object& objs, vec3d& start,light &l,vec3d &normal,int &seed) {

	float b = 0;
	bool shadow = false;

	vec3d toL = normalise(sub( l.pos,start));
	curandState s;	
	
	
	for (int j = 0; j < 10; j++) {
	
		vec3d P = cross(toL, vec3d({ 0,1,0 }));
		
	//	if (P.x == 0 && P.y == 0 && P.z == 0) {
	//		P.x = 1;
	//	}

		vec3d toEdge = normalise(sub(add(l.pos,multiply(P,l.size)),start));
		float angle = cosf((dotproduct(toL, toEdge)) * 2);
	
		float _z =  (float)j/10* (1.0f - angle) + angle;
		float phi = (float)j/10 * 2.f * 3.1415f;
/*       curand_init(seed += j, 0, 0, &s);
  
		float _z =  curand_uniform(&s) * (1.0f - angle) + angle;
		curand_init(seed+=j, 0, 0, &s);

		float phi = curand_uniform(&s) * 2.f * 3.1415f;*/ 

		float x = sqrtf(1.f - _z * _z) * cosf(phi);
		float y = sqrtf(1.f - _z * _z) * sinf(phi);

		vec3d axis = normalise(cross(vec3d({ 0, 0, 1 }), normalise(toL)));
		float nAngle = acosf(dotproduct(normalise(toL), vec3d({ 0,0,1 })));

		vec3d new_dir = normalise(sub(l.pos, multiply(rotate(nAngle, axis), vec3d({x,y,_z}))));
		ray light_ray({ start,new_dir });
	   
		shadow = false;

		//check for triangle intersection

		for (int j = 0; j < objs.mesh1->bvhbox_count; j++)
		{
			float temp;

			if (objs.mesh1->d_box[j].bvhbox->intersect(light_ray, temp)) {


				for (int i = 0; i < objs.mesh1->d_box[j].length; i++)
				{
					float t, u, v;

					if (objs.mesh1->rayIntersect(light_ray, objs.mesh1->d_tri_arr[objs.mesh1->d_box[j].d_indexes[i]], t, u, v))
					{
						shadow = true;
						break;
						//		return true;
					}
				}
				if (shadow) {
					break;
				}
			}
		}

	if(!shadow)
	//checkfor sphere intersection
	for (int i = 0; i < objs.sphere_count; i++)
	{
		float t;
		if (objs.d_spheres[i].intersect(light_ray, t))
		{
			shadow = true;
			break;
	//		return true;
		}
	}
	if (!shadow)
	//check for plane intersection
	for (int i = 0; i < objs.plane_count; i++)
	{
		float t;

		if (objs.d_planes[i].intersect(light_ray, t))
		{
			shadow = true;
			break;
		//	return true;
		}
	 }
	if (!shadow)
		//check for plane intersection
		for (int i = 0; i < objs.cube_count; i++)
		{
			float t;

			if (objs.d_cubes[i].intersect(light_ray, t))
			{
				shadow = true;
				break;
				//	return true;
			}
		}
	if (!shadow) {
	    	b += 0.1;
       }
	}
	float a = dotproduct(normal, toL);
	b *= a > 0 ? a : 0;
	return b;
}

__global__
void globalilumnation(object &obj, ray &reflect_ray,float &r, float &g, float &b) {
	
	float nv, nu, n_t;
	int hit_type, hit_index;



//	if (castRay(objs, reflect_ray, hit_type, hit_index, n_t, nu, nv)) {
/*
		float r_tx, r_ty;
		vec3d r_normal;

		if (hit_type == 0) {
			r_tx = ((1 - nu - nv) * objs.mesh1->d_tri_arr[hit_index].vt[0].u) + (nu * objs.mesh1->d_tri_arr[hit_index].vt[1].u) + (nv * objs.mesh1->d_tri_arr[hit_index].vt[2].u);  //(int)(((nu + 1.0f) / 2.0f) * maxX)-1;
			r_ty = ((1 - nu - nv) * objs.mesh1->d_tri_arr[hit_index].vt[0].v) + (nu * objs.mesh1->d_tri_arr[hit_index].vt[1].v) + (nv * objs.mesh1->d_tri_arr[hit_index].vt[2].v);//((float)maxY -((nv + 1.0f) / 2.0f) * maxY)-1;

			if (objs.mesh1->has_normals) {
				r_normal = add(add(multiply(objs.mesh1->d_tri_arr[hit_index].vecNormal[0], (1 - nu - nv)), multiply(objs.mesh1->d_tri_arr[hit_index].vecNormal[1], nu)), multiply(objs.mesh1->d_tri_arr[hit_index].vecNormal[2], nv));
				r_normal = normalise(r_normal);
			}
			else {
				r_normal = objs.mesh1->d_tri_arr[hit_index].normal;
			}
		}
		if (hit_type == 1) {

			vec3d hit_point = add(reflect_ray.Org, multiply(reflect_ray.Dir, n_t));
			r_normal = sub(hit_point, objs.d_spheres[hit_index].orgin);
			r_normal = normalise(r_normal);
			//calc the texture vertices
			r_tx = (1 + atan2(r_normal.z, r_normal.x) / 3.1415) * 0.5;
			r_ty = acosf(r_normal.y) / 3.1415;
		}
		if (hit_type == 2) {

			vec3d hit_point = add(reflect_ray.Org, multiply(reflect_ray.Dir, n_t));
			r_normal = sub(hit_point, objs.d_spheres[hit_index].orgin);
			r_normal = normalise(r_normal);
			//calc the texture vertices
			r_tx = 0.5;//(1 + atan2(r_normal.z, r_normal.x) / 3.1415) * 0.5;
			r_ty = 0.5;//acosf(r_normal.y) / 3.1415;
		}
		int r_index = (int)(r_ty * maxY) * maxX + (int)(r_tx * maxX);

		r *= objs.texture->rBuff->data[r_index];
		g *= objs.texture->gBuff->data[r_index];
		b *= objs.texture->bBuff->data[r_index];

		reflect_ray.Org = add(r_normal, add(reflect_ray.Org, multiply(reflect_ray.Dir, n_t)));
		reflect_ray.Dir = reflect(reflect_ray.Dir, r_normal);

	}
	//the ray didnt intersect with any object so set the color to the sky
	else {
		float temp_r, temp_g, temp_b;

		sky.getFColor(reflect_ray, temp_r, temp_g, temp_b);

		r *= objs.mat->reflectivness * temp_r;
		g *= objs.mat->reflectivness * temp_g;
		b *= objs.mat->reflectivness * temp_b;

		break;*/
//	}

}

__global__ 
void rayTrace(unsigned int* pixels, int width, int height, float aspect, object& objs, light* lights, int light_size, camera cam, skybox& sky) {

	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int x = threadIdx.x + blockIdx.x * blockDim.x;

	if (y >= height || x >= width)
		return;


		float dx = aspect * (2 * (x + 0.5) / (float)width) - 1 ;
		float dy = aspect * (2 * (y + 0.5) / (float)height)*((float)height/width) - 1;
	
		float nu, nv, n_t;

	vec3d eyePos({0,0,(-1/aspect)});
	vec3d dir = vec3d({ dx,dy,0 });
	ray cam_ray(add(eyePos,cam.Org),cam.rotateDir(normalise(sub(dir,eyePos)),cam.Camyaw,cam.Campitch));
    
	int hit_index;
	int hit_type;
	vec3d new_org;
	vec3d normal;
	float tx, ty;

	//if no ray intersect set the color of the skybox 
	if (castRay(objs, cam_ray, hit_type, hit_index, n_t, nu, nv,new_org,normal,tx,ty)) {


		int maxX = objs.texture->width;
		int maxY = objs.texture->height;

		vec3d reflect_dir = reflect(cam_ray.Dir, normal);
		vec3d start_O = add(multiply(normal,0.00001),new_org);
		vec3d obj_normal = normal;

		ray reflect_ray(new_org, reflect_dir);
		bool hit = false;

		int c_index = (int)(ty * maxY) * maxX + (int)(tx * maxX);

		float r = objs.texture->rBuff->data[c_index], g = objs.texture->gBuff->data[c_index], b = objs.texture->bBuff->data[c_index];
	
		//global illumination color contribution  values
		float Gr, Gg, Gb;
//		globalilumnation(objs,cam_ray,Gr,Gg,Gb);

		int seed = threadIdx.y + blockIdx.x * blockIdx.y + threadIdx.x*3;
	
		float fr = 0, fg = 0, fb = 0;
		//calc light value
	    for (int i = 0; i < light_size; i++)
		{
			
			bool shadow = false;
			seed*= seed;

				float brightness =  castLightRay(objs, start_O, lights[i],obj_normal,seed);
		
				fr += brightness * lights[i].r *r;
				fg += brightness * lights[i].g *g;
				fb += brightness * lights[i].b *b;

		}


		

		pixels[y * width + x] = rgbToInt(fr * 254, fg * 254, fb * 254);
		return;
   }
	   float r, g, b;
	   sky.getFColor(cam_ray, r, g, b);

	pixels[y * width + x] = rgbToInt(r*254,g*254,b*254);
	return;
}

int light_size = 3;
int tx =8, ty = 8;
light* lights;
camera cam({ 4,3,10 }, { 0, 0,1},0.f);

int lightByteSize = sizeof(float) * 37*light_size;

object* objs = new object();
skybox* Skybox = new skybox("C:\\Users\\Leon\\Downloads\\sky_box.jpg",10000);
float aspect = tan((90 * 0.5 * 3.1415) / 180);
float yawY = 0,yawX = 0;

void onStart() {

	objs->loadMesh("C:\\Users\\Leon\\Downloads\\skull2.obj", "C:\\Users\\Leon\\Downloads\\wood.jpg", material(0,0,0));

	light m_light({ 20,20,20 }, 20, 1, 0, 0);
	light b_light({ 0,20,-20 }, 20, 0, 0, 1);
	light c_light({ 0,20,0 }, 20, 0, 1, 0);

	lights = new light[3]{ m_light ,b_light,c_light};
	//cam.Camyaw = -80;
}

void checkKey() {

	vec3d point = multiply(vec3d({cam.Dir.x,0,cam.Dir.z }), 01);
	vec3d pointside = multiply(vec3d({ 1,0,0 }), 0.1);
	vec3d pointUp = multiply(vec3d({ 0,1,0 }), 0.1);


	if (GetKeyState('W') & 0x8000) {
		cam.Org.z -= 0.1;
	}
	
	if (GetKeyState('S') & 0x8000) {
		cam.Org.z += 0.1;
	}
	
	
	if (GetKeyState('A') & 0x8000) {
		cam.Org.x += 0.1;
	}
	
	if (GetKeyState('D') & 0x8000) {
		cam.Org.x -= 0.1;

	}
	if (GetKeyState('E') & 0x8000) {
		cam.Camyaw += 1;
	}
	if (GetKeyState('Q') & 0x8000) {
		cam.Camyaw -= 1;
	}
	if (GetKeyState('Z') & 0x8000) {
		cam.Campitch += 1;
	}

	if (GetKeyState('C') & 0x8000) {
		cam.Campitch -= 1;
	}
	if (GetKeyState(VK_SHIFT) & 0x8000) {
		cam.Org = add(cam.Org, pointUp);
	}
	if (GetKeyState(VK_SPACE) & 0x8000) {
		cam.Org = sub(cam.Org, pointUp);
	}
}


void update() {

	checkKey();
	//Skybox->box->orgin = cam.Org;

	unsigned int* pixels;
	light* d_lights;


	int width = getScreenWidth(), height = getScreenHeight();
	int pixelSize = width * height * sizeof(unsigned int);
	cam.aspect = (float)height / width;

	checkCudaErrors(cudaMallocManaged((void**)&pixels, pixelSize));
	checkCudaErrors(cudaMalloc((void**)&d_lights, lightByteSize));
	
	checkCudaErrors(cudaMemcpy(d_lights, lights, lightByteSize, cudaMemcpyHostToDevice));

	dim3 blocks(width / tx + 1, height / ty + 1);
	dim3 threads(tx, ty);

	rayTrace << <blocks, threads >> > (pixels, width, height, aspect, *objs, d_lights, light_size, cam,*Skybox);

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

    setPixelBuff(pixels);
	checkCudaErrors(cudaFree(pixels));
	checkCudaErrors(cudaFree(d_lights));
	
}
/*
int main() {
	onStart();
	while (true) {
update();
}
}*/ 
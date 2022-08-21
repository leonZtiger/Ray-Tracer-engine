#include "cuda_runtime.h"
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
	v2.x = v1.x * mat.mat[0][0] + v1.y * mat.mat[0][1] + v1.z * mat.mat[0][2];
	v2.y = v1.x * mat.mat[1][0] + v1.y * mat.mat[1][1] + v1.z * mat.mat[1][2];
	v2.z = v1.x * mat.mat[2][0] + v1.y * mat.mat[2][1] + v1.z * mat.mat[2][2];

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
		
		 w = normalise(sub(pos,dir));
		 u = normalise(cross(vec3d({ 0,1,0 }),w));
		 v =  cross(w,u);
        
		viewPlaneW = tan(90*3.1415/180);
		viewPlaneH = viewPlaneW * tan(90*3.1415/180);
		
		lower_left = sub(sub(sub(pos,multiply(u,viewPlaneW)),multiply(v,viewPlaneH)),w);
		horizontal = multiply(multiply(u, viewPlaneW),2);
		vertical = multiply(multiply(v, viewPlaneH), 2);

		this->Org = pos;
		this->Dir = dir;
		this->lens = apature / 2;
	}
	void lookAt() {

		w = normalise(sub(Org, Dir));
		u = normalise(cross(vec3d({ 0,1,0 }),w ));
		v = cross(w,u);

		viewPlaneW = tan((90 * 3.1415 / 180)/2);
		viewPlaneH = viewPlaneW * aspect;

		lower_left = sub(sub(sub(Org, multiply(u, viewPlaneW)), multiply(v, viewPlaneH)), w);
		horizontal = multiply(multiply(u, viewPlaneW), 2);
		vertical = multiply(multiply(v, viewPlaneH), 2);
	}
	__device__
	vec3d Rand_disk() {
		
		curandState s;
		curand_init(Org.x,0,0,&s);

		vec3d p = sub(multiply(vec3d({(float)curand_uniform_double(&s),(float)curand_uniform_double(&s),0}), 2), vec3d({1,1,0}));

		if (!dotproduct(p, p) >= 1) {
			return p;
		}
	}

	matrix lookAt(vec3d &from, vec3d &to) {
		vec3d temp = { 0,1,0 };

		vec3d viewDir = sub(to, from);
		u = normalise(multiply(viewDir, temp));
		v = normalise( multiply(u, viewDir));


		vec3d forward = normalise(sub(to, from));
		vec3d right = cross(normalise(temp),forward);
		vec3d up = cross(forward,right);
		
		matrix camToWorld;

		camToWorld.mat[0][0] = right.x;
		camToWorld.mat[0][1] = right.y;
		camToWorld.mat[0][2] = right.z;
		camToWorld.mat[1][0] = up.x;
		camToWorld.mat[1][1] = up.y;
		camToWorld.mat[1][2] = up.z;
		camToWorld.mat[2][0] = forward.x;
		camToWorld.mat[2][1] = forward.y;
		camToWorld.mat[2][2] = forward.z;

		camToWorld.mat[3][0] = from.x;
		camToWorld.mat[3][1] = from.y;
		camToWorld.mat[3][2] = from.z;

		return camToWorld;
	}
	__device__
	vec3d rotateDir(vec3d &vec) {
		float yawRad = yaw * (3.1415 / 180);
		float pitchRad = pitch * (3.1415 / 180);

		float y = vec.y * cosf(pitchRad) - vec.z * sinf(pitchRad);
		float z = vec.y * sinf(pitchRad) + vec.z * cosf(pitchRad);
		float x = vec.x * cosf(yawRad) + z * sinf(yawRad);
		z = -vec.x * sinf(yawRad) + z * cosf(yawRad);

		return { x,y,z };
	}


	vec3d w,u,v ;
	float viewPlaneW;
	float aspect;
	float viewPlaneH;
	float viewPlaneB;
	float lens;
	float yaw, pitch;
	vec3d lower_left;
	vec3d horizontal;
	vec3d vertical;
};


class shape : public memManager
{

public:

	    virtual int getBytes() { return 0; }
	
	    __device__
		virtual bool intersects(ray& cam_ray, float& t) { return true; };

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

		t0 = t0 > t1 ? t1 : t0;

		if (t0 < 0) {
			t0 = t1;
			if (t0 < 0)
				return false;
		}
		t = t0;
		return true;

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
		if (denom > 0.0001) {
			vec3d pl0 = sub(cam_ray.Org,orgin);
			t = dotproduct(pl0, normal);
			return (t >= 0);
		}
		return false;
	}

	bool reflective = false;
	vec3d normal;

};
__device__
bool planeIntersect(ray& cam_ray,plane p, float& t) {

	float denom = dotproduct(p.normal, cam_ray.Dir);
	if (denom >= 0.0001) {
		vec3d pl0 = sub(p.orgin, cam_ray.Org);
		t = dotproduct(pl0, p.normal);
		return (t >= 0);
	}
	return false;
}
class cube {

public:

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


class mesh :public memManager {

public:
    
	
	
	triangle *d_tri_arr;
	triangle* h_tri_arr;

    int poly_count;
	bool has_normals = false;

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
                if (vn.size() != 0 && vt.size() > 1) {
				if (SlashCount(line) >= 8) {
               
					
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
					tris.push_back({ p[f[0] - 1],p[f[2] - 1] ,p[f[3] - 1] , normal,vn[pvn[0] - 1],vn[pvn[2] - 1],vn[pvn[3] - 1] ,vt[pvt[0] - 1],vt[pvt[2] - 1] ,vt[pvt[3] - 1] });

					

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

		for (int i = 0; i < poly_count; i++) {
			tris[i].points[0].z -= 12;
			tris[i].points[1].z -= 12;
			tris[i].points[2].z -= 12;

	/*		tris[i].points[0].x -= 3;
			tris[i].points[1].x -= 3;
			tris[i].points[2].x -= 3;*/

			tris[i].points[0].y -= 3;
			tris[i].points[1].y -= 3;
			tris[i].points[2].y -= 3;

			h_tri_arr[i] = tris[i];
		}
		tris.clear();
		return;
	}
    
	
	
	void allocMem() {
		int size = getBytes();
		checkCudaErrors(cudaMallocManaged((void**)&d_tri_arr, size));
		memcpy(d_tri_arr, h_tri_arr, size);
	}

	__device__ __host__
	int getBytes() {
		return ((sizeof(float) * 3*7)+(sizeof(float)*2*3)) * poly_count;
	}
	
	/*__device__
	bool intersect(ray& ray, int &color, float &n_t, vec3d &normal) {
		
		 float u, v,t;
		 float n_u, n_v;
		 
		 for (size_t i = 0; i < poly_count; i++)
		 {
			 if (triangleIntersect(ray, triangles[i], t, u, v)) {
				 if (t < n_t) {
					 n_u = u;
					 n_v = v;
					 n_t = t;
					 normal = triangles[i].normal;
				}
			}
		 }
		 if (n_t != INFINITY) {
		 color = rgbToInt(n_u*255,n_v*255,(1-n_u-n_v)*255);
		 return true;
		 }
		 return false;
	 }*/
	
	

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

		int x = (int)((1 + atan2(normal.z, normal.x) / 3.1415) * 0.5 * skyboxTex->width);
		int y = (int)(acosf(normal.y) / 3.1415 * skyboxTex->height);

		int index = y * skyboxTex->width + x;

		r = skyboxTex->rBuff->data[index];
		g = skyboxTex->gBuff->data[index];
		b = skyboxTex->bBuff->data[index];

	}

protected:
	
	sprite* skyboxTex;
	sphere* box;

};


class object : public memManager {

public:
	object() {}

	void loadMesh(string file, string tex, material mat) {

		mesh1 = new mesh(file);
		mesh1->allocMem();
		s1 = new sphere[sphere_count];
		planes = new plane({ 0,-50,0 }, { 0,1, 0 });

		for (int i = 0; i < sphere_count; i++) {
			s1[i] = sphere({ (float)(rand() % 100) / 10 ,0 ,(float)(rand() % 100) / 10 }, (float)(rand() % 100) / 100);
		}



		texture = new sprite(tex);
		this->mat = new material(mat);
		sphereAllocMem();
		planeAllocMem();
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

	int sphere_count = 5, plane_count = 1;
	int depth = 3;
	sphere* s1;
	sphere* d_spheres;
	plane* planes;
	plane* d_planes;
	mesh* mesh1;
	sprite* texture;
	material* mat;
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

__device__
vec3d reflect(vec3d &I, vec3d &N) {
	return sub(I,multiply(multiply(N,dotproduct(I,N)),2));
}
__global__
void setup_world() {

}
__device__
bool lightIntersect(ray& ray, triangle& tri) {

	vec3d edge1 = sub(tri.points[1], tri.points[0]);
	vec3d edge2 = sub(tri.points[2], tri.points[0]);
	vec3d h, s, q;
	float a, f,u,v;

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

	float t = f * dotproduct(edge2, q);
	if (t > 0.0000001) {
		return true;
	}
	else {
		return false;
	}

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
__device__
bool castRay(object &objs,ray &cam_ray,int &hit_type,int &hit_index,float &nt,float&nu,float &nv,vec3d & new_org,vec3d &normal,float & tx, float &ty) {
	
	nt = INFINITY;
	//check for triangle intersection
	for (int i = 0; i < objs.mesh1->poly_count; i++)
	{
		float t, u, v;

		if (rayIntersect(cam_ray, objs.mesh1->d_tri_arr[i], t, u, v))
		{

			if (t < nt)
			{
				nt = t;
				nv = v;
				nu = u;
				hit_index = i;
				hit_type = 0;
			}
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
			normal = normalise(normal);

			//texture vertices
			tx = 0.5; //hit_point.x;
			ty = 0.5;//hit_point.z* mod(1.f);

		}
		return true;
	}
		
	
	return false;
}
__global__
void globalilumnation(object objs, ray reflect_ray) {
	
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
	ray cam_ray(add(eyePos,cam.Org),cam.rotateDir(normalise(sub(dir,eyePos))));
    
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

		ray reflect_ray(new_org, reflect_dir);
		bool hit = false;

		int c_index = (int)(ty * maxY) * maxX + (int)(tx * maxX);

		float dr = 1, dg = 1, db = 1;
		float r = objs.texture->rBuff->data[c_index], g = objs.texture->gBuff->data[c_index], b = objs.texture->bBuff->data[c_index];

		//calculate the global ilumination the object recives
		// im jusing a routhness value to ditermen how many rays should be cast
		for (int depth = 0; depth < 4; depth++) {
		//	int rays = objs.mat->roughness*10+1;
        //	globalilumnation <<<1, rays>>> (objs,reflect_ray);
		//	cudaDeviceSynchronize();

			if (castRay(objs, reflect_ray, hit_type, hit_index, n_t, nu, nv,new_org, normal, tx, ty)) {
				c_index = (int)(ty * maxY) * maxX + (int)(tx * maxX);
				r *= objs.texture->rBuff->data[c_index], g *= objs.texture->gBuff->data[c_index], b *= objs.texture->bBuff->data[c_index];
				
				reflect_ray.Dir = reflect(reflect_ray.Dir,normal);
				reflect_ray.Org = new_org;
			}
			else {
			    sky.getFColor(reflect_ray,r,g,b);
				break;
			}
		}
	
		//calc light value
		for (int i = 0; i < light_size; i++)
		{

			vec3d new_dir = normalise(multiply(sub(new_org, lights[i].pos), -1));

			ray light_ray({ new_org,new_dir });

			bool shadow = false;

			for (int i = 0; i < objs.mesh1->poly_count; i++)
			{

				if (lightIntersect(light_ray, objs.mesh1->d_tri_arr[i]))
				{
					shadow = true;
					break;
				}
			}
			if (!shadow) {
				float angle = dotproduct(normal, light_ray.Dir);
				float brightness = 0.f < angle ? angle : 0.1f;

				dr *= 0.8 * brightness * r;// * lights[i].r*r;
				dg *= 0.8 * brightness * g;//* lights[i].g*g;
				db *= 0.8 * brightness * b; ////* b * lights[i].b*b;
			}
			else {
				dr *= 0.5 * r;
				dg *= 0.5 * g;
				db *= 0.5 * b;
			}
		}

		pixels[y * width + x] = rgbToInt(dr * 255, dg * 255, db * 255);

		return;
   }
	   float r, g, b;
	   sky.getFColor(cam_ray, r, g, b);

	pixels[y * width + x] = rgbToInt(r*255,g*255,b*255);
	return;
}

int light_size = 1;
int tx =8, ty = 8;
light* lights;
camera cam({ 0,0,-10 }, { 0, 0,0},0.f);

int lightByteSize = sizeof(float) * 10;

object* objs = new object();
skybox* Skybox = new skybox("C:\\Users\\Leon\\Downloads\\sky_box.jpg",10000);
float aspect = tan((90 * 0.5 * 3.1415) / 180);
float yawY = 0,yawX = 0;

void onStart() {

	objs->loadMesh("C:\\Users\\Leon\\Downloads\\.obj", "C:\\Users\\Leon\\Downloads\\marble.png", material(0,0,0));

	light m_light({ 10,30,-10 }, 1, 1, 1, 1);
	light b_light({ 10,30,-10 }, 1, 0, 0.5, 0.5);
	lights = new light[1] {m_light};

}

void checkKey() {

	vec3d point = multiply(vec3d({0,0,-1 }), 0.1);
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
		cam.yaw += 1;
	}
	if (GetKeyState('Q') & 0x8000) {
		cam.yaw -= 1;
	}
	if (GetKeyState('Z') & 0x8000) {
		cam.pitch += 1;
	}

	if (GetKeyState('C') & 0x8000) {
		cam.pitch -= 1;
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

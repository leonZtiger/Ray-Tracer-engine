#include "kernel.cuh"
#include "window.h"

bool running = true;

using namespace std;



struct Render_State {
	void* buffmemory;
	int width, height;

	BITMAPINFO buff_bitmap;
};
int width, height;
Render_State render;

LRESULT CALLBACK window_callback(HWND hwnd, UINT msg, WPARAM wparam, LPARAM lParam)
{
	LRESULT result = 0;
	//ta hand om olika callbacks
	switch (msg)
	{
	case WM_CLOSE:
	case WM_DESTROY: {
		running = false;
	} break;
	case WM_SIZE: {
		RECT rect;
		GetClientRect(hwnd, &rect);
		width = (rect.right - rect.left);
		height = (rect.bottom - rect.top);
		render.width = width * 0.5;
		render.height = height * 0.5;

		if (render.buffmemory)
			VirtualFree(render.buffmemory, 0, MEM_RELEASE);

		render.buffmemory = VirtualAlloc(0, (render.width * render.height * sizeof(unsigned int)), MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
		render.buff_bitmap.bmiHeader.biSize = sizeof(render.buff_bitmap.bmiHeader);
		render.buff_bitmap.bmiHeader.biWidth = render.width;
		render.buff_bitmap.bmiHeader.biHeight = render.height;
		render.buff_bitmap.bmiHeader.biPlanes = 1;
		render.buff_bitmap.bmiHeader.biBitCount = 32;
		render.buff_bitmap.bmiHeader.biCompression = BI_RGB;
	    
	}
	default: {
		result = DefWindowProc(hwnd, msg, wparam, lParam);
		break;
	}
	}
	return result;
}

int WINAPI wWinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, PWSTR pCmdLine, int nCmdShow) {

	//skappa window klass
	const char class_name[] = "ray tracer";
	WNDCLASS window_class = { 0 };
	window_class.style = CS_HREDRAW | CS_VREDRAW;
	window_class.lpszClassName = class_name;
	window_class.lpfnWndProc = window_callback;
	//rega klassen
	RegisterClass(&window_class);
	//skappa ett fönster
	HWND window = CreateWindow(window_class.lpszClassName, class_name, WS_OVERLAPPEDWINDOW | WS_VISIBLE, CW_USEDEFAULT, CW_USEDEFAULT, 1000, 1000, 0, 0, hInstance, 0);
	HDC hdc = GetDC(window);
	
	onStart();
	
	while (running) {
		MSG msg;
		while (PeekMessage(&msg, window, 0, 0, PM_REMOVE)) {
			TranslateMessage(&msg);
			DispatchMessage(&msg);
		}
		
	update();
		//render
		StretchDIBits(hdc, 0, 0,width, height, 0, 0, render.width, render.height, render.buffmemory, &render.buff_bitmap, DIB_RGB_COLORS, SRCCOPY);
	}
}

int getScreenHeight() {
	return render.height;
}
int getScreenWidth() {
	return render.width;
}
void setScreen( int *pixels) {

}
void drawPixel(int x, int y, int color) {
	x = make_inbound(0, render.width - 1, x);
	y = make_inbound(0, render.height - 1, y);

	unsigned int* pixel = (unsigned int*)render.buffmemory + x + y * (render.width);
	*pixel = color;
}
void Set_Background() {
	unsigned int* pixel = (unsigned int*)render.buffmemory;
	for (int y = 0; y < getScreenHeight(); y++) {
		for (int x = 0; x < getScreenWidth(); x++) {
			*pixel++ = y * x / (x + 1);
		}
	}
}
void Clear_Screen(unsigned int color) {
	unsigned int* pixel = (unsigned int*)render.buffmemory;
	for (int y = 0; y < getScreenHeight(); y++) {
		for (int x = 0; x < getScreenWidth(); x++) {
			*pixel++ = color;
		}
	}
}
int getBuffSize() {
	return sizeof(render.buffmemory);
}
int make_inbound(int min, int max, int val) {
	if (val > max) {
		return max;
	}
	if (val < min) {
		return min;
	}
	return val;
}
void setPixelBuff(unsigned int *pixels) {
	memcpy(render.buffmemory, pixels, sizeof(unsigned int)*render.width*render.height);
}
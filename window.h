#pragma once
#include <windows.h>
#include<string>



int getScreenWidth();
int getScreenHeight();

void drawPixel(int x, int y, int color);
void setPixelBuff(unsigned int *pixels);
void Set_Background();
void Clear_Screen(unsigned int color);
int make_inbound(int min, int max, int val);
int getBuffSize();
void setScreen(int *pixels);
#include<string>
#include<Windows.h>
#include<vector>

class key {

public:

	key(char key) {
		key_code = key;
	}
	bool State() {
	return	state;
	}
	void checkKey() {
		if (GetKeyState(key_code) & 0x8000) {
			state = true;
		}
		else {
			state = false;
		}
	}

private:
	char key_code;
	bool state;
};

class keyhandler {

public:
	keyhandler(char *keysC) {

		for (size_t i = 0; i < sizeof(keysC); i++)
		{
			keys.push_back( key(keysC[i]));
		}
	}
	void addKeyListener() {

	}
	bool run() {

		while (true) {

		}
	}
  	

private:
	std::vector<key> keys;
};
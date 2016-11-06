#include <vector>
#ifndef IMG_H_
#define IMG_H_

class img {
private:
	std::size_t width;
	std::size_t height;
	std::vector<unsigned int> massiv;
	unsigned int *GPU_img;
	void copy_to_GPU();
public:
	void filter();
	img(const char *filename, std::size_t width, std::size_t height);
	img(std::size_t width, std::size_t height,unsigned int *img_mass);
	void save(const char *filename);
	int operator() (int x, int y);
	unsigned int* to_massiv();
	std::size_t get_width(){return width;}
	std::size_t get_height(){return height;}
	std::size_t get_size(){return massiv.size()*sizeof(unsigned int);}
	unsigned int* get_copy_in_GPU();
	~img();
};

#endif /* IMG_H_ */

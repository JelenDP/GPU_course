
__kernel void jump_flood(global int* data,
                         global int* seed,
                         unsigned int w,
                         unsigned int h)
{
	int x = get_global_id(0);
    int y = get_global_id(1);

    //int w = get_global_size(0);
    //int h = get_global_size(1);

}
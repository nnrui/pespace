import taichi as ti
ti.init(arch=ti.gpu, offline_cache=False)


init = ti.field(ti.i8, shape=(2, 3))

# init[0, 0] = 1
# print(init)

@ti.kernel
def assign_a_cell():
    init[0, 0] = ti.cast(1, ti.i8)
assign_a_cell()
print(init)
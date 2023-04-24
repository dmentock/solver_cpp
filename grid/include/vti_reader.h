#ifndef VTI_READER_H
#define VTI_READER_H

struct VtiReader {
    public:
        virtual int* read_vti_material_data(const char* filename, int (&cells)[3], double (&geom_size)[3], double (&origin)[3]);
};
#endif // VTI_READER_H

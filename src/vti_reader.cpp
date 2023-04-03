#include <mpi.h>

#include <vtkSmartPointer.h>
#include <vtkXMLImageDataReader.h>
#include <vtkImageData.h>
#include <vtkCellData.h>
#include <vtkDataArray.h>

#include "vti_reader.h"

int* VtiReader::read_vti_material_data(const char* filename, int (&cells)[3], double (&geomSize)[3], double (&origin)[3]){
  vtkSmartPointer<vtkXMLImageDataReader> reader =
    vtkSmartPointer<vtkXMLImageDataReader>::New();
  reader->SetFileName(filename);
  reader->Update();
  vtkSmartPointer<vtkImageData> imageData = reader->GetOutput();
  vtkSmartPointer<vtkCellData> cellData = imageData->GetCellData();
  vtkSmartPointer<vtkDataArray> dataArray = cellData->GetArray("material");

  double geomSize_[6];
  imageData->GetBounds(geomSize_);
  for (int i = 0; i < 3; ++i) geomSize[i] = geomSize_[1+i*2];
  double* origin_ = imageData->GetOrigin();
  for (int i = 0; i < 3; ++i) origin[i] = origin_[i];
  int* cells_;
  imageData->GetDimensions(cells_);
  for (int i = 0; i < 3; ++i) cells[i] = cells_[i]-1;
  int* grid = new int[cells[0]*cells[1]*cells[2]];
  for (int i = 0; i < cells[0]; i++) {
    for (int j = 0; j < cells[1]; j++) {
      for (int k = 0; k < cells[2]; k++) {
        double value = dataArray->GetComponent(k + cells[2] * (j + cells[1] * i), 0);
        if (value < 1){
          std::cerr << "Material ID < 1" << std::endl;
          MPI_Abort(MPI_COMM_WORLD, 1);
        }
        std::cout  << value <<", ";
        grid[k + cells[1] * j + cells[0] * cells[1] * i] = (int)value;
      }
    }
  }
  return grid;
}



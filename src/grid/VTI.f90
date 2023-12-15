!--------------------------------------------------------------------------------------------------
!> @author Martin Diehl, KU Leuven
!> @brief Read data from image files of the visualization toolkit.

! TODO: use minimal c++ parser like tinyxml (https://github.com/leethomason/tinyxml2)
! look into managing dependencies with conan (https://conan.io/)
!--------------------------------------------------------------------------------------------------
module VTI
  use prec
  use zlib
  use base64
  use IO
  use iso_c_binding

  implicit none(type,external)
  private

  public :: &
    VTI_readDataset_int_c, &
    VTI_readDataset_int, &
    VTI_readDataset_real, &
    VTI_readCellsSizeOrigin

contains

!--------------------------------------------------------------------------------------------------
!> @brief Read integer dataset from a VTK image data (*.vti) file.
!> @details https://vtk.org/Wiki/VTK_XML_Formats
!--------------------------------------------------------------------------------------------------
function VTI_readDataset_int(fileContent,label) result(dataset)

  character(len=*), intent(in) :: &
    label, &
    fileContent
  integer, dimension(:), allocatable :: &
    dataset

  character(len=:), allocatable :: dataType, headerType, base64Str
  logical :: compressed


  call VTI_readDataset_raw(base64Str,dataType,headerType,compressed, &
                           fileContent,label)
  dataset = as_Int(base64Str,headerType,compressed,dataType)

end function VTI_readDataset_int


!--------------------------------------------------------------------------------------------------
!> @brief Read real dataset from a VTK image data (*.vti) file.
!> @details https://vtk.org/Wiki/VTK_XML_Formats
!--------------------------------------------------------------------------------------------------
function VTI_readDataset_real(fileContent,label) result(dataset)

  character(len=*), intent(in) :: &
    label, &
    fileContent
  real(pREAL),  dimension(:), allocatable :: &
    dataset

  character(len=:), allocatable :: dataType, headerType, base64Str
  logical :: compressed


  call VTI_readDataset_raw(base64Str,dataType,headerType,compressed, &
                           fileContent,label)
  dataset = as_real(base64Str,headerType,compressed,dataType)

end function VTI_readDataset_real


!--------------------------------------------------------------------------------------------------
!> @brief Read dataset as raw data (base64 string) from a VTK image data (*.vti) file.
!> @details https://vtk.org/Wiki/VTK_XML_Formats
!--------------------------------------------------------------------------------------------------
subroutine VTI_readDataset_raw(base64Str,dataType,headerType,compressed, &
                                  fileContent,label)

  character(len=*), intent(in) :: &
    label, &
    fileContent
  character(len=:), allocatable, intent(out) :: dataType, headerType, base64Str
  logical, intent(out) :: compressed

  logical :: inFile, inImage
  integer(pI64) :: &
    startPos, endPos, &
    s


  inFile = .false.
  inImage = .false.
  startPos = 1_pI64
  outer: do while (startPos < len(fileContent,kind=pI64))
    endPos = startPos + index(fileContent(startPos:),IO_EOL,kind=pI64) - 2_pI64
    if (endPos < startPos) endPos = len(fileContent,kind=pI64)                                      ! end of file without new line

    if (.not. inFile) then
      if (index(fileContent(startPos:endPos),'<VTKFile',kind=pI64) /= 0_pI64) then
        inFile = .true.
        if (.not. fileFormatOk(fileContent(startPos:endPos))) call IO_error(error_ID = 844, ext_msg='file format')
        headerType = merge('UInt64','UInt32',getXMLValue(fileContent(startPos:endPos),'header_type')=='UInt64')
        compressed  = getXMLValue(fileContent(startPos:endPos),'compressor') == 'vtkZLibDataCompressor'
      end if
    else
      if (.not. inImage) then
        if (index(fileContent(startPos:endPos),'<ImageData',kind=pI64) /= 0_pI64) then
          inImage = .true.
        end if
      else
        if (index(fileContent(startPos:endPos),'<CellData',kind=pI64) /= 0_pI64) then
          do while (index(fileContent(startPos:endPos),'</CellData>',kind=pI64) == 0_pI64)
            if (index(fileContent(startPos:endPos),'<DataArray',kind=pI64) /= 0_pI64 .and. &
                 getXMLValue(fileContent(startPos:endPos),'Name') == label ) then

              if (getXMLValue(fileContent(startPos:endPos),'format') /= 'binary') &
                call IO_error(error_ID = 844, ext_msg='format ('//label//')')
              dataType = getXMLValue(fileContent(startPos:endPos),'type')

              startPos = endPos + 2_pI64
              endPos  = startPos + index(fileContent(startPos:),IO_EOL,kind=pI64) - 2_pI64
              s = startPos + verify(fileContent(startPos:endPos),IO_WHITESPACE,kind=pI64) -1_pI64   ! start (no leading whitespace)
              base64Str = fileContent(s:endPos)
              exit outer
            end if
            startPos = endPos + 2_pI64
            endPos = startPos + index(fileContent(startPos:),IO_EOL,kind=pI64) - 2_pI64
          end do
        end if
      end if
    end if

    startPos = endPos + 2_pI64

  end do outer

  if (.not. allocated(base64Str)) call IO_error(error_ID = 844, ext_msg='dataset "'//label//'" not found')

end subroutine VTI_readDataset_raw


!--------------------------------------------------------------------------------------------------
!> @brief Read cells, size, and origin of an VTK image data (*.vti) file.
!> @details https://vtk.org/Wiki/VTK_XML_Formats
!--------------------------------------------------------------------------------------------------
subroutine VTI_readCellsSizeOrigin(fileContent_c,string_length,cells,geomSize,origin) bind(C, name="f_VTI_readCellsSizeOrigin")

  character(kind=c_char),          intent(in) :: &
    fileContent_c
  integer(c_int), value :: string_length
  character(len=string_length) :: &
    fileContent
  integer(c_int), dimension(3), intent(out) :: &
    cells                                                                                           ! # of cells (across all processes!)
  real(c_double), dimension(3), intent(out) :: &
    geomSize, &                                                                                       ! size (across all processes!)
    origin                                                                                          ! origin (across all processes!)

  character(len=:), allocatable :: headerType
  logical :: inFile, inImage, compressed
  integer(pI64) :: &
    startPos, endPos


  fileContent = c_str_to_f_str(fileContent_c, string_length)
  cells = -1
  geomSize = -1.0_pREAL

  inFile = .false.
  inImage = .false.
  startPos = 1_pI64
  outer: do while (startPos < len(fileContent,kind=pI64))
    endPos = startPos + index(fileContent(startPos:),IO_EOL,kind=pI64) - 2_pI64
    if (endPos < startPos) endPos = len(fileContent,kind=pI64)                                      ! end of file without new line

    if (.not. inFile) then
      if (index(fileContent(startPos:endPos),'<VTKFile',kind=pI64) /= 0_pI64) then
        inFile = .true.
        if (.not. fileFormatOk(fileContent(startPos:endPos))) call IO_error(error_ID = 844, ext_msg='file format')
        headerType = merge('UInt64','UInt32',getXMLValue(fileContent(startPos:endPos),'header_type')=='UInt64')
        compressed  = getXMLValue(fileContent(startPos:endPos),'compressor') == 'vtkZLibDataCompressor'
      end if
    else
      if (.not. inImage) then
        if (index(fileContent(startPos:endPos),'<ImageData',kind=pI64) /= 0_pI64) then
          inImage = .true.
          call cellsSizeOrigin(cells,geomSize,origin,fileContent(startPos:endPos))
          exit outer
        end if
      end if
    end if

    startPos = endPos + 2_pI64

  end do outer

  if (any(geomSize<=0))                 call IO_error(error_ID = 844, ext_msg='size')
  if (any(cells<1))                     call IO_error(error_ID = 844, ext_msg='cells')
  print *, "HAHAHAHAHA", geomSize
  print *, "cells", cells

end subroutine VTI_readCellsSizeOrigin


!--------------------------------------------------------------------------------------------------
!> @brief Determine size and origin from coordinates.
!--------------------------------------------------------------------------------------------------
subroutine cellsSizeOrigin(c,s,o,header)

  integer, dimension(3),     intent(out) :: c
  real(pREAL), dimension(3), intent(out) :: s,o
  character(len=*),          intent(in) :: header

  character(len=:), allocatable :: temp
  real(pREAL), dimension(3) :: delta
  integer :: i


  temp = getXMLValue(header,'Direction')
  if (temp /= '1 0 0 0 1 0 0 0 1' .and. temp /= '') &                                               ! https://discourse.vtk.org/t/vti-specification/6526
    call IO_error(error_ID = 844, ext_msg = 'coordinate order')

  temp = getXMLValue(header,'WholeExtent')
  if (any([(IO_intValue(temp,IO_strPos(temp),i),i=1,5,2)] /= 0)) &
    call IO_error(error_ID = 844, ext_msg = 'coordinate start')
  c = [(IO_intValue(temp,IO_strPos(temp),i),i=2,6,2)]

  temp = getXMLValue(header,'Spacing')
  delta = [(IO_realValue(temp,IO_strPos(temp),i),i=1,3)]
  s = delta * real(c,pREAL)

  temp = getXMLValue(header,'Origin')
  o = [(IO_realValue(temp,IO_strPos(temp),i),i=1,3)]

end subroutine cellsSizeOrigin


!--------------------------------------------------------------------------------------------------
!> @brief Interpret Base64 string in vtk XML file as integer of default kind.
!--------------------------------------------------------------------------------------------------
function as_Int(base64Str,headerType,compressed,dataType)

  character(len=*), intent(in) :: base64Str, &                                                      ! base64 encoded string
                                  headerType, &                                                     ! header type (UInt32 or Uint64)
                                  dataType                                                          ! data type (Int32, Int64, Float32, Float64)
  logical,          intent(in) :: compressed                                                        ! indicate whether data is zlib compressed

  integer, dimension(:), allocatable :: as_Int


  select case(dataType)
    case('Int32')
      as_Int = int(prec_bytesToC_INT32_T(asBytes(base64Str,headerType,compressed)))
    case('Int64')
      as_Int = int(prec_bytesToC_INT64_T(asBytes(base64Str,headerType,compressed)))
    case('Float32')
      as_Int = int(prec_bytesToC_FLOAT  (asBytes(base64Str,headerType,compressed)))
    case('Float64')
      as_Int = int(prec_bytesToC_DOUBLE (asBytes(base64Str,headerType,compressed)))
    case default
      call IO_error(844,ext_msg='unknown data type: '//trim(dataType))
  end select

end function as_Int


!--------------------------------------------------------------------------------------------------
!> @brief Interpret Base64 string in vtk XML file as real of kind pREAL.
!--------------------------------------------------------------------------------------------------
function as_real(base64Str,headerType,compressed,dataType)

  character(len=*), intent(in) :: base64Str, &                                                      ! base64 encoded string
                                  headerType, &                                                     ! header type (UInt32 or Uint64)
                                  dataType                                                          ! data type (Int32, Int64, Float32, Float64)
  logical,          intent(in) :: compressed                                                        ! indicate whether data is zlib compressed

  real(pREAL), dimension(:), allocatable :: as_real


  select case(dataType)
    case('Int32')
      as_real = real(prec_bytesToC_INT32_T(asBytes(base64Str,headerType,compressed)),pREAL)
    case('Int64')
      as_real = real(prec_bytesToC_INT64_T(asBytes(base64Str,headerType,compressed)),pREAL)
    case('Float32')
      as_real = real(prec_bytesToC_FLOAT  (asBytes(base64Str,headerType,compressed)),pREAL)
    case('Float64')
      as_real = real(prec_bytesToC_DOUBLE (asBytes(base64Str,headerType,compressed)),pREAL)
    case default
      call IO_error(844,ext_msg='unknown data type: '//trim(dataType))
  end select

end function as_real


!--------------------------------------------------------------------------------------------------
!> @brief Interpret Base64 string in vtk XML file as bytes.
!--------------------------------------------------------------------------------------------------
function asBytes(base64Str,headerType,compressed) result(bytes)

  character(len=*), intent(in) :: base64Str, &                                                      ! base64 encoded string
                                  headerType                                                        ! header type (UInt32 or Uint64)
  logical,          intent(in) :: compressed                                                        ! indicate whether data is zlib compressed

  integer(C_SIGNED_CHAR), dimension(:), allocatable :: bytes


  if (compressed) then
    bytes = asBytes_compressed(base64Str,headerType)
  else
    bytes = asBytes_uncompressed(base64Str,headerType)
  end if

end function asBytes


!--------------------------------------------------------------------------------------------------
!> @brief Interpret compressed Base64 string in vtk XML file as bytes.
!> @details A compressed Base64 string consists of a header block and a data block
! [#blocks/#u-size/#p-size/#c-size-1/#c-size-2/.../#c-size-#blocks][DATA-1/DATA-2...]
! #blocks = Number of blocks
! #u-size = Block size before compression
! #p-size = Size of last partial block (zero if it not needed)
! #c-size-i = Size in bytes of block i after compression
!--------------------------------------------------------------------------------------------------
function asBytes_compressed(base64Str,headerType) result(bytes)

  character(len=*), intent(in) :: base64Str, &                                                      ! base64 encoded string
                                  headerType                                                        ! header type (UInt32 or Uint64)
  integer(C_SIGNED_CHAR), dimension(:), allocatable :: bytes

  integer(C_SIGNED_CHAR), dimension(:), allocatable :: bytes_inflated
  integer(pI64), dimension(:), allocatable :: temp, size_inflated, size_deflated
  integer(pI64) :: headerLen, nBlock, b,s,e


  if      (headerType == 'UInt32') then
    temp = int(prec_bytesToC_INT32_T(base64_to_bytes(base64Str(:base64_nChar(4_pI64)))),pI64)
    nBlock = int(temp(1),pI64)
    headerLen = 4_pI64 * (3_pI64 + nBlock)
    temp = int(prec_bytesToC_INT32_T(base64_to_bytes(base64Str(:base64_nChar(headerLen)))),pI64)
  else if (headerType == 'UInt64') then
    temp = int(prec_bytesToC_INT64_T(base64_to_bytes(base64Str(:base64_nChar(8_pI64)))),pI64)
    nBlock = int(temp(1),pI64)
    headerLen = 8_pI64 * (3_pI64 + nBlock)
    temp = int(prec_bytesToC_INT64_T(base64_to_bytes(base64Str(:base64_nChar(headerLen)))),pI64)
  end if

  allocate(size_inflated(nBlock),source=temp(2))
  size_inflated(nBlock) = merge(temp(3),temp(2),temp(3)/=0_pI64)
  size_deflated = temp(4:)
  bytes_inflated = base64_to_bytes(base64Str(base64_nChar(headerLen)+1_pI64:))

  allocate(bytes(sum(size_inflated)))
  e = 0_pI64
  do b = 1, nBlock
    s = e + 1_pI64
    e = s + size_deflated(b) - 1_pI64
    bytes(sum(size_inflated(:b-1))+1_pI64:sum(size_inflated(:b))) = zlib_inflate(bytes_inflated(s:e),size_inflated(b))
  end do

end function asBytes_compressed


!--------------------------------------------------------------------------------------------------
!> @brief Interprete uncompressed Base64 string in vtk XML file as bytes.
!> @details An uncompressed Base64 string consists of N headers blocks and a N data blocks
![#bytes-1/DATA-1][#bytes-2/DATA-2]...
!--------------------------------------------------------------------------------------------------
function asBytes_uncompressed(base64Str,headerType) result(bytes)

  character(len=*), intent(in) :: base64Str, &                                                      ! base64 encoded string
                                  headerType                                                        ! header type (UInt32 or Uint64)
  integer(C_SIGNED_CHAR), dimension(:), allocatable :: bytes

  integer(pI64) :: s
  integer(pI64), dimension(1) :: nByte


  allocate(bytes(0))

  s=0_pI64
  if      (headerType == 'UInt32') then
    do while(s+base64_nChar(4_pI64)<(len(base64Str,pI64)))
      nByte = int(prec_bytesToC_INT32_T(base64_to_bytes(base64Str(s+1_pI64:s+base64_nChar(4_pI64)))),pI64)
      bytes = [bytes,base64_to_bytes(base64Str(s+1_pI64:s+base64_nChar(4_pI64+nByte(1))),5_pI64)]
      s = s + base64_nChar(4_pI64+nByte(1))
    end do
  else if (headerType == 'UInt64') then
    do while(s+base64_nChar(8_pI64)<(len(base64Str,pI64)))
      nByte = int(prec_bytesToC_INT64_T(base64_to_bytes(base64Str(s+1_pI64:s+base64_nChar(8_pI64)))),pI64)
      bytes = [bytes,base64_to_bytes(base64Str(s+1_pI64:s+base64_nChar(8_pI64+nByte(1))),9_pI64)]
      s = s + base64_nChar(8_pI64+nByte(1))
    end do
  end if

end function asBytes_uncompressed


!--------------------------------------------------------------------------------------------------
!> @brief Get XML string value for given key.
!--------------------------------------------------------------------------------------------------
pure function getXMLValue(line,key)

  character(len=*), intent(in)  :: line, key
  character(len=:), allocatable :: getXMLValue

  integer :: s,e
#ifdef __INTEL_COMPILER
  character :: q
#endif


  s = index(line," "//key,back=.true.)
  if (s==0) then
    getXMLValue = ''
  else
    e = s + 1 + scan(line(s+1:),"'"//'"')
    if (scan(line(s:e-2),'=') == 0) then
      getXMLValue = ''
    else
      s = e
!https://community.intel.com/t5/Intel-Fortran-Compiler/ICE-for-merge-with-strings/m-p/1207204#M151657
#ifdef __INTEL_COMPILER
      q = line(s-1:s-1)
      e = s + index(line(s:),q) - 1
#else
      e = s + index(line(s:),merge("'",'"',line(s-1:s-1)=="'")) - 1
#endif
      getXMLValue = line(s:e-1)
    end if
  end if

end function getXMLValue


!--------------------------------------------------------------------------------------------------
!> @brief Check for supported file format variants.
!--------------------------------------------------------------------------------------------------
pure function fileFormatOk(line)

  character(len=*),intent(in) :: line
  logical :: fileFormatOk


  fileFormatOk = getXMLValue(line,'type')       == 'ImageData' .and. &
                 getXMLValue(line,'byte_order') == 'LittleEndian' .and. &
                 getXMLValue(line,'compressor') /= 'vtkLZ4DataCompressor' .and. &
                 getXMLValue(line,'compressor') /= 'vtkLZMADataCompressor'

end function fileFormatOk

function c_str_to_f_str(c_string, string_length) result(f_string)
  integer(c_int), value :: string_length
  character(kind=c_char), dimension(string_length), intent(in) :: c_string
  character(len=string_length) :: f_string

  integer :: i, str_length

  str_length = 0
  do i = 1, string_length
    if (c_string(i) == c_null_char) exit
    f_string(i:i) = c_string(i)
    str_length = str_length + 1
  end do

end function c_str_to_f_str

!--------------------------------------------------------------------------------------------------
!> @brief Read integer dataset from a VTK image data (*.vti) file.
!> @details https://vtk.org/Wiki/VTK_XML_Formats
!--------------------------------------------------------------------------------------------------
subroutine VTI_readDataset_int_c(fileContent_c, string_length, label_c, label_length, dataset, dataset_size) &
  bind(C, name="f_VTI_readDataset_int")
  character(kind=c_char),          intent(in) :: &
    fileContent_c, label_c
  integer(c_int), value,  intent(in) :: string_length, label_length
  character(len=label_length) :: &
    label
  character(len=string_length) :: &
    fileContent
  integer :: &
    dataset_size
  integer, dimension(dataset_size) :: &
    dataset

  character(len=:), allocatable :: dataType, headerType, base64Str
  logical :: compressed

  fileContent = c_str_to_f_str(fileContent_c, string_length)
  label = c_str_to_f_str(label_c, label_length)
  call VTI_readDataset_raw(base64Str,dataType,headerType,compressed, &
                           fileContent,label)
  dataset = as_Int(base64Str,headerType,compressed,dataType) + 1
  dataset_size = size(dataset)

end subroutine VTI_readDataset_int_c

end module VTI

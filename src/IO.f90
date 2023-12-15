!--------------------------------------------------------------------------------------------------
!> @author Franz Roters, Max-Planck-Institut für Eisenforschung GmbH
!> @author Philip Eisenlohr, Max-Planck-Institut für Eisenforschung GmbH
!> @author Christoph Kords, Max-Planck-Institut für Eisenforschung GmbH
!> @author Martin Diehl, Max-Planck-Institut für Eisenforschung GmbH
!> @brief  input/output functions
!--------------------------------------------------------------------------------------------------
module IO
  use, intrinsic :: ISO_fortran_env, only: &
    IO_STDOUT => OUTPUT_UNIT, &
    IO_STDERR => ERROR_UNIT

  use prec
  use constants
  use misc

  implicit none(type,external)
  private

  character(len=*), parameter, public :: &
    IO_WHITESPACE = achar(44)//achar(32)//achar(9)//achar(10)//achar(13), &                         !< whitespace characters
    IO_QUOTES  = "'"//'"'
  character, parameter, public :: &
    IO_EOL = LF, &                                                                                  !< end of line character
    IO_COMMENT = '#'

  public :: &
    IO_init, &
    IO_selfTest, &
    IO_read, &
    IO_readlines, &
    IO_isBlank, &
    IO_wrapLines, &
    IO_strPos, &
    IO_strValue, &
    IO_intValue, &
    IO_realValue, &
    IO_lc, &
    IO_rmComment, &
    IO_glueDiffering, &
    IO_intAsStr, &
    IO_strAsInt, &
    IO_strAsReal, &
    IO_strAsBool, &
    IO_error, &
    IO_warning, &
    IO_STDOUT

contains


!--------------------------------------------------------------------------------------------------
!> @brief Do self test.
!--------------------------------------------------------------------------------------------------
subroutine IO_init()

  print'(/,1x,a)', '<<<+-  IO init  -+>>>'; flush(IO_STDOUT)

  call IO_selfTest()

end subroutine IO_init


!--------------------------------------------------------------------------------------------------
!> @brief Read ASCII file and split at EOL.
!--------------------------------------------------------------------------------------------------
function IO_readlines(fileName) result(fileContent)

  character(len=*),       intent(in)                :: fileName
  character(len=pSTRLEN), dimension(:), allocatable :: fileContent                                  !< file content, separated per lines

  character(len=pSTRLEN)                            :: line
  character(len=:),                     allocatable :: rawData
  integer ::  &
    startPos, endPos, &
    N_lines, &                                                                                      !< # lines in file
    l
  logical :: warned


  rawData = IO_read(fileName)

  N_lines = count([(rawData(l:l) == IO_EOL,l=1,len(rawData))])
  allocate(fileContent(N_lines))

!--------------------------------------------------------------------------------------------------
! split raw data at end of line
  warned = .false.
  startPos = 1
  l = 1
  do while (l <= N_lines)
    endPos = startPos + scan(rawData(startPos:),IO_EOL) - 2
    if (endPos - startPos > pSTRLEN-1) then
      line = rawData(startPos:startPos+pSTRLEN-1)
      if (.not. warned) then
        call IO_warning(207,trim(fileName),label1='line',ID1=l)
        warned = .true.
      end if
    else
      line = rawData(startPos:endpos)
    end if
    startPos = endPos + 2                                                                           ! jump to next line start

    fileContent(l) = trim(line)//''
    l = l + 1
  end do

end function IO_readlines


!--------------------------------------------------------------------------------------------------
!> @brief Read ASCII file.
!> @details Proper Unix style (LF line endings and LF at EOF) is ensured.
!--------------------------------------------------------------------------------------------------
function IO_read(fileName) result(fileContent)

  character(len=*),  intent(in) :: fileName
  character(len=:), allocatable :: fileContent

  integer ::  &
    fileUnit, &
    myStat
  integer(pI64) ::  &
    fileLength


  inquire(file = fileName, size=fileLength)
  open(newunit=fileUnit, file=fileName, access='stream',&
       status='old', position='rewind', action='read',iostat=myStat)
  if (myStat /= 0) call IO_error(100,trim(fileName))
  allocate(character(len=fileLength)::fileContent)
  if (fileLength==0) then
    close(fileUnit)
    return
  end if

  read(fileUnit,iostat=myStat) fileContent
  if (myStat /= 0) call IO_error(102,trim(fileName))
  close(fileUnit)

  if (index(fileContent,CR//LF,kind=pI64) /= 0)     fileContent = CRLF2LF(fileContent)
  if (fileContent(fileLength:fileLength) /= IO_EOL) fileContent = fileContent//IO_EOL               ! ensure EOL@EOF

end function IO_read


!--------------------------------------------------------------------------------------------------
!> @brief Identifiy strings without content.
!--------------------------------------------------------------------------------------------------
logical pure function IO_isBlank(str)

  character(len=*), intent(in) :: str                                                               !< string to check for content

  integer :: posNonBlank


  posNonBlank = verify(str,IO_WHITESPACE)
  IO_isBlank = posNonBlank == 0 .or. posNonBlank == scan(str,IO_COMMENT)

end function IO_isBlank


!--------------------------------------------------------------------------------------------------
!> @brief Insert EOL at separator trying to keep line length below limit.
!--------------------------------------------------------------------------------------------------
function IO_wrapLines(str,separator,filler,length)

  character(len=*),    intent(in) :: str                                                            !< string to split
  character, optional, intent(in) :: separator                                                      !< line breaks are possible after this character, defaults to ','
  character(len=*), optional, intent(in) :: filler                                                  !< character(s) to insert after line break, defaults to none
  integer,   optional, intent(in) :: length                                                         !< (soft) line limit, defaults to 80
  character(len=:), allocatable :: IO_wrapLines

  integer, dimension(:), allocatable :: pos_sep, pos_split
  integer :: i,s,e


  i = index(str,misc_optional(separator,','))
  if (i == 0) then
    IO_wrapLines = str
  else
    pos_sep = [0]
    s = i
    do while (i /= 0 .and. s < len(str))
      pos_sep = [pos_sep,s]
      i = index(str(s+1:),misc_optional(separator,','))
      s = s + i
    end do
    pos_sep = [pos_sep,len(str)]

    pos_split = emptyIntArray
    s = 1
    e = 2
    IO_wrapLines = ''
    do while (e < size(pos_sep))
      if (pos_sep(e+1) - pos_sep(s) >= misc_optional(length,80)) then
        IO_wrapLines = IO_wrapLines//adjustl(str(pos_sep(s)+1:pos_sep(e)))//IO_EOL//misc_optional(filler,'')
        s = e
      end if
      e = e + 1
    end do
    IO_wrapLines = IO_wrapLines//adjustl(str(pos_sep(s)+1:))
  end if

end function IO_wrapLines


!--------------------------------------------------------------------------------------------------
!> @brief Locate all whitespace-separated chunks in given string and returns array containing
!! number them and the left/right position to be used by IO_xxxVal.
!! Array size is dynamically adjusted to number of chunks found in string
!! IMPORTANT: first element contains number of chunks!
!--------------------------------------------------------------------------------------------------
pure function IO_strPos(str)

  character(len=*),                  intent(in) :: str                                              !< string in which chunk positions are searched for
  integer, dimension(:), allocatable            :: IO_strPos

  integer                      :: left, right


  allocate(IO_strPos(1), source=0)
  right = 0

  do while (verify(str(right+1:),IO_WHITESPACE)>0)
    left  = right + verify(str(right+1:),IO_WHITESPACE)
    right = left + scan(str(left:),IO_WHITESPACE) - 2
    if ( str(left:left) == IO_COMMENT) exit
    IO_strPos = [IO_strPos,left,right]
    IO_strPos(1) = IO_strPos(1)+1
    endOfStr: if (right < left) then
      IO_strPos(IO_strPos(1)*2+1) = len_trim(str)
      exit
    end if endOfStr
  end do

end function IO_strPos


!--------------------------------------------------------------------------------------------------
!> @brief Read string value at myChunk from string.
!--------------------------------------------------------------------------------------------------
function IO_strValue(str,chunkPos,myChunk)

  character(len=*),             intent(in) :: str                                                   !< raw input with known start and end of each chunk
  integer,   dimension(:),      intent(in) :: chunkPos                                              !< positions of start and end of each tag/chunk in given string
  integer,                      intent(in) :: myChunk                                               !< position number of desired chunk
  character(len=:), allocatable            :: IO_strValue

  validChunk: if (myChunk > chunkPos(1) .or. myChunk < 1) then
    IO_strValue = ''
    call IO_error(110,'IO_strValue: "'//trim(str)//'"',label1='chunk',ID1=myChunk)
  else validChunk
    IO_strValue = str(chunkPos(myChunk*2):chunkPos(myChunk*2+1))
  end if validChunk

end function IO_strValue


!--------------------------------------------------------------------------------------------------
!> @brief Read integer value at myChunk from string.
!--------------------------------------------------------------------------------------------------
integer function IO_intValue(str,chunkPos,myChunk)

  character(len=*),      intent(in) :: str                                                          !< raw input with known start and end of each chunk
  integer, dimension(:), intent(in) :: chunkPos                                                     !< positions of start and end of each tag/chunk in given string
  integer,               intent(in) :: myChunk                                                      !< position number of desired chunk

  IO_intValue = IO_strAsInt(IO_strValue(str,chunkPos,myChunk))

end function IO_intValue


!--------------------------------------------------------------------------------------------------
!> @brief Read real value at myChunk from string.
!--------------------------------------------------------------------------------------------------
real(pREAL) function IO_realValue(str,chunkPos,myChunk)

  character(len=*),        intent(in) :: str                                                        !< raw input with known start and end of each chunk
  integer,   dimension(:), intent(in) :: chunkPos                                                   !< positions of start and end of each tag/chunk in given string
  integer,                 intent(in) :: myChunk                                                    !< position number of desired chunk

  IO_realValue = IO_strAsReal(IO_strValue(str,chunkPos,myChunk))

end function IO_realValue


!--------------------------------------------------------------------------------------------------
!> @brief Convert characters in string to lower case.
!--------------------------------------------------------------------------------------------------
pure function IO_lc(str)

  character(len=*), intent(in) :: str                                                               !< string to convert
  character(len=len(str))   :: IO_lc

  integer :: i,n


  do i = 1,len(str)
    n = index(UPPER,str(i:i))
    if (n==0) then
      IO_lc(i:i) = str(i:i)
    else
      IO_lc(i:i) = LOWER(n:n)
    end if
  end do

end function IO_lc


!--------------------------------------------------------------------------------------------------
! @brief Remove comments (characters beyond '#') and trailing space.
! ToDo: Discuss name (the trim aspect is not clear)
!--------------------------------------------------------------------------------------------------
function IO_rmComment(line)

  character(len=*), intent(in)  :: line
  character(len=:), allocatable :: IO_rmComment

  integer :: split


  split = index(line,IO_COMMENT)

  if (split == 0) then
    IO_rmComment = trim(line)
  else
    IO_rmComment = trim(line(:split-1))
  end if

end function IO_rmComment


!--------------------------------------------------------------------------------------------------
! @brief Return first (with glued on second if they differ).
!--------------------------------------------------------------------------------------------------
function IO_glueDiffering(first,second,glue)

  character(len=*),           intent(in)  :: first
  character(len=*),           intent(in)  :: second
  character(len=*), optional, intent(in)  :: glue
  character(len=:), allocatable :: IO_glueDiffering

  character(len=:), allocatable           :: glue_


  glue_ = misc_optional(glue,'<--')
  IO_glueDiffering = trim(first)
  if (trim(first) /= trim(second)) IO_glueDiffering = IO_glueDiffering//' '//trim(glue_)//' '//trim(second)

end function IO_glueDiffering


!--------------------------------------------------------------------------------------------------
!> @brief Return given int value as string.
!--------------------------------------------------------------------------------------------------
function IO_intAsStr(i)

  integer, intent(in)            :: i
  character(len=:), allocatable  :: IO_intAsStr


  allocate(character(len=merge(2,1,i<0) + floor(log10(real(abs(merge(1,i,i==0))))))::IO_intAsStr)
  write(IO_intAsStr,'(i0)') i

end function IO_intAsStr


!--------------------------------------------------------------------------------------------------
!> @brief Return integer value from given string.
!--------------------------------------------------------------------------------------------------
integer function IO_strAsInt(str)

  character(len=*), intent(in) :: str                                                               !< string for conversion to int value

  integer                      :: readStatus
  character(len=*), parameter  :: VALIDCHARS = '0123456789+- '


  valid: if (verify(str,VALIDCHARS) == 0) then
    read(str,*,iostat=readStatus) IO_strAsInt
    if (readStatus /= 0) call IO_error(111,str)
  else valid
    IO_strAsInt = 0
    call IO_error(111,str)
  end if valid

end function IO_strAsInt


!--------------------------------------------------------------------------------------------------
!> @brief Return real value from given string.
!--------------------------------------------------------------------------------------------------
real(pREAL) function IO_strAsReal(str)

  character(len=*), intent(in) :: str                                                               !< string for conversion to real value

  integer                      :: readStatus
  character(len=*), parameter  :: VALIDCHARS = '0123456789eE.+- '


  valid: if (verify(str,VALIDCHARS) == 0) then
    read(str,*,iostat=readStatus) IO_strAsReal
    if (readStatus /= 0) call IO_error(112,str)
  else valid
    IO_strAsReal = 0.0_pREAL
    call IO_error(112,str)
  end if valid

end function IO_strAsReal


!--------------------------------------------------------------------------------------------------
!> @brief Return logical value from given string.
!--------------------------------------------------------------------------------------------------
logical function IO_strAsBool(str)

  character(len=*), intent(in) :: str                                                               !< string for conversion to int value


  if     (trim(adjustl(str)) == 'True' .or.  trim(adjustl(str)) == 'true') then
    IO_strAsBool = .true.
  elseif (trim(adjustl(str)) == 'False' .or. trim(adjustl(str)) == 'false') then
    IO_strAsBool = .false.
  else
    IO_strAsBool = .false.
    call IO_error(113,str)
  end if

end function IO_strAsBool



!--------------------------------------------------------------------------------------------------
!> @brief Write error statements and terminate the run with exit #9xxx.
!--------------------------------------------------------------------------------------------------
subroutine IO_error(error_ID,ext_msg,label1,ID1,label2,ID2)

  integer,                    intent(in) :: error_ID
  character(len=*), optional, intent(in) :: ext_msg,label1,label2
  integer,          optional, intent(in) :: ID1,ID2

  external                      :: quit
  character(len=:), allocatable :: msg

  select case (error_ID)

!--------------------------------------------------------------------------------------------------
! file handling errors
    case (100)
      msg = 'could not open file:'
    case (102)
      msg = 'could not read file:'

!--------------------------------------------------------------------------------------------------
! file parsing errors
    case (110)
      msg = 'invalid chunk selected'
    case (111)
      msg = 'invalid character for int:'
    case (112)
      msg = 'invalid character for real:'
    case (113)
      msg = 'invalid character for logical:'
    case (114)
      msg = 'cannot decode base64 string:'

!--------------------------------------------------------------------------------------------------
! lattice error messages
    case (130)
      msg = 'unknown lattice structure encountered'
    case (131)
      msg = 'hex lattice structure with invalid c/a ratio'
    case (132)
      msg = 'invalid parameters for transformation'
    case (134)
      msg = 'negative lattice parameter'
    case (135)
      msg = 'zero entry on stiffness diagonal'
    case (137)
      msg = 'not defined for lattice structure'
    case (138)
      msg = 'not enough interaction parameters given'

!--------------------------------------------------------------------------------------------------
! errors related to the parsing of material.yaml
    case (140)
      msg = 'key not found'
    case (141)
      msg = 'number of chunks in string differs'
    case (142)
      msg = 'empty list'
    case (143)
      msg = 'no value found for key'
    case (144)
      msg = 'negative number systems requested'
    case (145)
      msg = 'too many systems requested'
    case (146)
      msg = 'number of values does not match'
    case (147)
      msg = 'V_e needs to be symmetric'
    case (148)
      msg = 'Nconstituents mismatch between homogenization and material'

!--------------------------------------------------------------------------------------------------
! material error messages and related messages in geometry
    case (150)
      msg = 'index out of bounds'
    case (153)
      msg = 'sum of phase fractions differs from 1'
    case (155)
      msg = 'material index out of bounds'
    case (180)
      msg = 'missing/invalid material definition'
    case (190)
      msg = 'unknown element type:'
    case (191)
      msg = 'mesh contains more than one element type'

!--------------------------------------------------------------------------------------------------
! plasticity error messages
    case (200)
      msg = 'unknown elasticity specified:'
    case (201)
      msg = 'unknown plasticity specified:'

    case (211)
      msg = 'material parameter out of bounds:'
    case (212)
      msg = 'nonlocal model not supported'

!--------------------------------------------------------------------------------------------------
! numerics error messages
    case (301)
      msg = 'numerics parameter out of bounds:'

!--------------------------------------------------------------------------------------------------
! math errors
    case (402)
      msg = 'invalid orientation specified'

!-------------------------------------------------------------------------------------------------
! homogenization errors
    case (500)
      msg = 'unknown homogenization specified'
    case (501)
      msg = 'homogenization description absent'

!--------------------------------------------------------------------------------------------------
! user errors
    case (600)
      msg = 'only one source entry allowed'
    case (603)
      msg = 'invalid data for table'
    case (610)
      msg = 'missing argument for option'
    case (611)
      msg = 'could not parse restart increment'
    case (612)
      msg = 'missing option'
    case (630)
      msg = 'JOBNAME must not contain any slashes'
    case (640)
      msg = 'invalid working directory'



!------------------------------------------------------------------------------------------------
! errors related to YAML data
    case (701)
      msg = 'incorrect indent/Null value not allowed'
    case (702)
      msg = 'invalid use of flow YAML'
    case (703)
      msg = 'invalid YAML'
    case (704)
      msg = 'space expected after a colon for <key>: <value> pair'
    case (705)
      msg = 'unsupported feature'
    case (706)
      msg = 'type mismatch in YAML data node'
    case (707)
      msg = 'abrupt end of file'
    case (708)
      msg = '"---" expected after YAML file header'
    case (709)
      msg = 'length mismatch'
    case (710)
      msg = 'closing quotation mark missing in string'
    case (711)
      msg = 'incorrect type'

!-------------------------------------------------------------------------------------------------
! errors related to the mesh solver
    case (821)
      msg = 'order not supported'

!-------------------------------------------------------------------------------------------------
! errors related to the grid solver
    case (831)
      msg = 'mask consistency violated in grid load case'
    case (833)
      msg = 'non-positive ratio for geometric progression'
    case (834)
      msg = 'negative time increment in grid load case'
    case (835)
      msg = 'non-positive increments in grid load case'
    case (836)
      msg = 'non-positive result frequency in grid load case'
    case (837)
      msg = 'incomplete loadcase'
    case (838)
      msg = 'mixed boundary conditions allow rotation'
    case (839)
      msg = 'non-positive restart frequency in grid load case'
    case (844)
      msg = 'invalid VTI file'
    case (891)
      msg = 'unknown solver type selected'
    case (892)
      msg = 'unknown filter type selected'
    case (894)
      msg = 'MPI error'

    case (950)
      msg = 'max number of cutbacks exceeded, terminating'

    case default
      error stop 'invalid error number'

  end select

  call panel('error',error_ID,msg, &
                     ext_msg=ext_msg, &
                     label1=label1,ID1=ID1, &
                     label2=label2,ID2=ID2)
  call quit(9000+error_ID)

end subroutine IO_error


!--------------------------------------------------------------------------------------------------
!> @brief Write warning statements.
!--------------------------------------------------------------------------------------------------
subroutine IO_warning(warning_ID,ext_msg,label1,ID1,label2,ID2)

  integer,                    intent(in) :: warning_ID
  character(len=*), optional, intent(in) :: ext_msg,label1,label2
  integer,          optional, intent(in) :: ID1,ID2

  character(len=:), allocatable :: msg


  select case (warning_ID)
    case (47)
      msg = 'invalid parameter for FFTW'
    case (207)
      msg = 'line truncated'
    case (600)
      msg = 'crystallite responds elastically'
    case (601)
      msg = 'stiffness close to zero'
    case (709)
      msg = 'read only the first document'

    case default
      error stop 'invalid warning number'
  end select

  call panel('warning',warning_ID,msg, &
                 ext_msg=ext_msg, &
                 label1=label1,ID1=ID1, &
                 label2=label2,ID2=ID2)

end subroutine IO_warning


!--------------------------------------------------------------------------------------------------
!> @brief Convert Windows (CRLF) to Unix (LF) line endings.
!--------------------------------------------------------------------------------------------------
pure function CRLF2LF(str)

  character(len=*), intent(in)  :: str
  character(len=:), allocatable :: CRLF2LF

  integer(pI64) :: c,n


  allocate(character(len=len_trim(str,pI64))::CRLF2LF)
  if (len(CRLF2LF,pI64) == 0) return

  n = 0_pI64
  do c=1_pI64, len_trim(str,pI64)
    CRLF2LF(c-n:c-n) = str(c:c)
    if (c == len_trim(str,pI64)) exit
    if (str(c:c+1_pI64) == CR//LF) n = n + 1_pI64
  end do

  CRLF2LF = CRLF2LF(:c-n)

end function CRLF2LF


!--------------------------------------------------------------------------------------------------
!> @brief Write statements to standard error.
!--------------------------------------------------------------------------------------------------
subroutine panel(paneltype,ID,msg,ext_msg,label1,ID1,label2,ID2)

  character(len=*),           intent(in) :: paneltype,msg
  character(len=*), optional, intent(in) :: ext_msg,label1,label2
  integer,                    intent(in) :: ID
  integer,          optional, intent(in) :: ID1,ID2

  character(len=pSTRLEN)                 :: formatString
  integer, parameter                     :: panelwidth = 69
  character(len=:), allocatable          :: msg_,ID_,msg1,msg2
  character(len=*), parameter            :: DIVIDER = repeat('─',panelwidth)


  if (.not. present(label1) .and.       present(ID1)) error stop 'missing label for value 1'
  if (.not. present(label2) .and.       present(ID2)) error stop 'missing label for value 2'

  ID_ = IO_intAsStr(ID)
  if (present(label1)) msg1 = label1
  if (present(label2)) msg2 = label2
  if (present(ID1)) msg1 = msg1//' '//IO_intAsStr(ID1)
  if (present(ID2)) msg2 = msg2//' '//IO_intAsStr(ID2)

  if (paneltype == 'error')   msg_ = achar(27)//'[31m'//trim(msg)//achar(27)//'[0m'
  if (paneltype == 'warning') msg_ = achar(27)//'[33m'//trim(msg)//achar(27)//'[0m'
  !$OMP CRITICAL (write2out)
  write(IO_STDERR,'(/,a)')                ' ┌'//DIVIDER//'┐'
  write(formatString,'(a,i2,a)') '(a,24x,a,1x,i0,',max(1,panelwidth-24-len_trim(paneltype)-1-len_trim(ID_)),'x,a)'
  write(IO_STDERR,formatString)          ' │',trim(paneltype),ID,                                   '│'
  write(IO_STDERR,'(a)')                  ' ├'//DIVIDER//'┤'
  write(formatString,'(a,i3.3,a,i3.3,a)') '(1x,a4,a',max(1,len_trim(msg_)),',',&
                                                     max(1,panelwidth+3-len_trim(msg)-4),'x,a)'
  write(IO_STDERR,formatString)            '│ ',trim(msg_),                                         '│'
  if (present(ext_msg)) then
    write(formatString,'(a,i3.3,a,i3.3,a)') '(1x,a4,a',max(1,len_trim(ext_msg)),',',&
                                                       max(1,panelwidth+3-len_trim(ext_msg)-4),'x,a)'
    write(IO_STDERR,formatString)          '│ ',trim(ext_msg),                                      '│'
  end if
  if (present(label1)) then
    write(formatString,'(a,i3.3,a,i3.3,a)') '(1x,a7,a',max(1,len_trim(msg1)),',',&
                                                       max(1,panelwidth+3-len_trim(msg1)-7),'x,a)'
    write(IO_STDERR,formatString)          '│ at ',trim(msg1),                                     '│'
  end if
  if (present(label2)) then
    write(formatString,'(a,i3.3,a,i3.3,a)') '(1x,a7,a',max(1,len_trim(msg2)),',',&
                                                       max(1,panelwidth+3-len_trim(msg2)-7),'x,a)'
    write(IO_STDERR,formatString)          '│ at ',trim(msg2),                                     '│'
  end if
  write(formatString,'(a,i2.2,a)') '(a,',max(1,panelwidth),'x,a)'
  write(IO_STDERR,formatString)          ' │',                                                     '│'
  write(IO_STDERR,'(a)')                  ' └'//DIVIDER//'┘'
  flush(IO_STDERR)
  !$OMP END CRITICAL (write2out)

end subroutine panel


!--------------------------------------------------------------------------------------------------
!> @brief Check correctness of some IO functions.
!--------------------------------------------------------------------------------------------------
subroutine IO_selfTest()

  integer, dimension(:), allocatable :: chunkPos
  character(len=:),      allocatable :: str,out


  if (dNeq(1.0_pREAL, IO_strAsReal('1.0')))          error stop 'IO_strAsReal'
  if (dNeq(1.0_pREAL, IO_strAsReal('1e0')))          error stop 'IO_strAsReal'
  if (dNeq(0.1_pREAL, IO_strAsReal('1e-1')))         error stop 'IO_strAsReal'
  if (dNeq(0.1_pREAL, IO_strAsReal('1.0e-1')))       error stop 'IO_strAsReal'
  if (dNeq(0.1_pREAL, IO_strAsReal('1.00e-1')))      error stop 'IO_strAsReal'
  if (dNeq(10._pREAL, IO_strAsReal(' 1.0e+1 ')))     error stop 'IO_strAsReal'

  if (3112019  /= IO_strAsInt( '3112019'))           error stop 'IO_strAsInt'
  if (3112019  /= IO_strAsInt(' 3112019'))           error stop 'IO_strAsInt'
  if (-3112019 /= IO_strAsInt('-3112019'))           error stop 'IO_strAsInt'
  if (3112019  /= IO_strAsInt('+3112019 '))          error stop 'IO_strAsInt'
  if (3112019  /= IO_strAsInt('03112019 '))          error stop 'IO_strAsInt'
  if (3112019  /= IO_strAsInt('+03112019'))          error stop 'IO_strAsInt'

  if (.not. IO_strAsBool(' true'))                   error stop 'IO_strAsBool'
  if (.not. IO_strAsBool(' True '))                  error stop 'IO_strAsBool'
  if (      IO_strAsBool(' false'))                  error stop 'IO_strAsBool'
  if (      IO_strAsBool('False'))                   error stop 'IO_strAsBool'

  if ('1234' /= IO_intAsStr(1234))                   error stop 'IO_intAsStr'
  if ('-12'  /= IO_intAsStr(-0012))                  error stop 'IO_intAsStr'

  if (any([1,1,1]     /= IO_strPos('a')))            error stop 'IO_strPos'
  if (any([2,2,3,5,5] /= IO_strPos(' aa b')))        error stop 'IO_strPos'

  str = ' 1.0 xxx'
  chunkPos = IO_strPos(str)
  if (dNeq(1.0_pREAL,IO_realValue(str,chunkPos,1)))  error stop 'IO_realValue'

  str = 'M 3112019 F'
  chunkPos = IO_strPos(str)
  if (3112019 /= IO_intValue(str,chunkPos,2))        error stop 'IO_intValue'

  if (CRLF2LF('') /= '')                             error stop 'CRLF2LF/0'
  if (CRLF2LF(LF)     /= LF)                         error stop 'CRLF2LF/1a'
  if (CRLF2LF(CR//LF) /= LF)                         error stop 'CRLF2LF/1b'
  if (CRLF2LF(' '//LF)     /= ' '//LF)               error stop 'CRLF2LF/2a'
  if (CRLF2LF(' '//CR//LF) /= ' '//LF)               error stop 'CRLF2LF/2b'
  if (CRLF2LF('A'//CR//LF//'B') /= 'A'//LF//'B')     error stop 'CRLF2LF/3'
  if (CRLF2LF('A'//CR//LF//'B'//CR//LF) /= &
              'A'//LF//'B'//LF)                      error stop 'CRLF2LF/4'
  if (CRLF2LF('A'//LF//CR//'B') /= 'A'//LF//CR//'B') error stop 'CRLF2LF/5'

  str='  ';        if (.not. IO_isBlank(str))        error stop 'IO_isBlank/1'
  str='  #isBlank';if (.not. IO_isBlank(str))        error stop 'IO_isBlank/2'
  str='  i#s';     if (      IO_isBlank(str))        error stop 'IO_isBlank/3'

  str='*(HiU!)3';if ('*(hiu!)3' /= IO_lc(str))       error stop 'IO_lc'

  str='#';out=IO_rmComment(str)
  if (out /= ''   .or. len(out) /= 0)                error stop 'IO_rmComment/1'
  str=' #';out=IO_rmComment(str)
  if (out /= ''   .or. len(out) /= 0)                error stop 'IO_rmComment/2'
  str=' # ';out=IO_rmComment(str)
  if (out /= ''   .or. len(out) /= 0)                error stop 'IO_rmComment/3'
  str=' # a';out=IO_rmComment(str)
  if (out /= ''   .or. len(out) /= 0)                error stop 'IO_rmComment/4'
  str=' a#';out=IO_rmComment(str)
  if (out /= ' a' .or. len(out) /= 2)                error stop 'IO_rmComment/5'
  str=' ab #';out=IO_rmComment(str)
  if (out /= ' ab'.or. len(out) /= 3)                error stop 'IO_rmComment/6'

  if ('abc, def' /= IO_wrapLines('abc, def')) &
                                                     error stop 'IO_wrapLines/1'
  if ('abc,'//IO_EOL//'def' /= IO_wrapLines('abc,def',length=3)) &
                                                     error stop 'IO_wrapLines/2'
  if ('abc,'//IO_EOL//'def' /= IO_wrapLines('abc,def',length=5)) &
                                                     error stop 'IO_wrapLines/3'
  if ('abc, def' /= IO_wrapLines('abc, def',length=3,separator='.')) &
                                                     error stop 'IO_wrapLines/4'
  if ('abc.'//IO_EOL//'def' /= IO_wrapLines('abc. def',length=3,separator='.')) &
                                                     error stop 'IO_wrapLines/5'
  if ('abc,'//IO_EOL//'defg,'//IO_EOL//'hij' /= IO_wrapLines('abc,defg,hij',length=4)) &
                                                     error stop 'IO_wrapLines/6'
  if ('abc,'//IO_EOL//'xxdefg,'//IO_EOL//'xxhij' /= IO_wrapLines('abc,defg, hij',filler='xx',length=4)) &
                                                     error stop 'IO_wrapLines/7'

end subroutine IO_selfTest

end module IO

module tensor_printer

    use prec

    implicit none(type,external)


    public :: p2r, p2c, p3r, p3c, p4r, p4c, p5r, p5c, p6r, p6c, p7r, p7c

contains

    subroutine p1r(label, tensor)
        character(len=*), intent(in) :: label
        real(pReal), intent(in) :: tensor(:)
        print *, label, shape(tensor), "vals:", tensor
    end subroutine p1r

    subroutine p1c(label, tensor)
        character(len=*), intent(in) :: label
        complex(pReal), intent(in) :: tensor(:)
        print *, label, shape(tensor), "vals:", tensor
    end subroutine p1c

    subroutine p2r(label, tensor)
        character(len=*), intent(in) :: label
        real(pReal), intent(in) :: tensor(:, :)
        print *, label, shape(tensor), "vals:", tensor
    end subroutine p2r

    subroutine p2c(label, tensor)
        character(len=*), intent(in) :: label
        complex(pReal), intent(in) :: tensor(:, :)
        print *, label, shape(tensor), "vals:", tensor
    end subroutine p2c

    subroutine p3r(label, tensor)
        character(len=*), intent(in) :: label
        real(pReal), intent(in) :: tensor(:, :, :)
        print *, label, shape(tensor), "vals:", tensor
    end subroutine p3r

    subroutine p3c(label, tensor)
        character(len=*), intent(in) :: label
        complex(pReal), intent(in) :: tensor(:, :, :)
        print *, label, shape(tensor), "vals:", tensor
    end subroutine p3c

    subroutine p4r(label, tensor)
        character(len=*), intent(in) :: label
        real(pReal), intent(in) :: tensor(:, :, :, :)
        print *, label, shape(tensor), "vals:", tensor
    end subroutine p4r

    subroutine p4c(label, tensor)
        character(len=*), intent(in) :: label
        complex(pReal), intent(in) :: tensor(:, :, :, :)
        print *, label, shape(tensor), "vals:", tensor
    end subroutine p4c

    subroutine p5r(label, tensor)
        character(len=*), intent(in) :: label
        real(pReal), intent(in) :: tensor(:, :, :, :, :)
        print *, label, shape(tensor), "vals:", tensor
    end subroutine p5r

    subroutine p5c(label, tensor)
        character(len=*), intent(in) :: label
        complex(pReal), intent(in) :: tensor(:, :, :, :, :)
        print *, label, shape(tensor), "vals:", tensor
    end subroutine p5c

    subroutine p6r(label, tensor)
        character(len=*), intent(in) :: label
        real(pReal), intent(in) :: tensor(:, :, :, :, :, :)
        print *, label, shape(tensor), "vals:", tensor
    end subroutine p6r

    subroutine p6c(label, tensor)
        character(len=*), intent(in) :: label
        complex(pReal), intent(in) :: tensor(:, :, :, :, :, :)
        print *, label, shape(tensor), "vals:", tensor
    end subroutine p6c

    subroutine p7r(label, tensor)
        character(len=*), intent(in) :: label
        real(pReal), intent(in) :: tensor(:, :, :, :, :, :, :)
        print *, label, shape(tensor), "vals:", tensor
    end subroutine p7r

    subroutine p7c(label, tensor)
        character(len=*), intent(in) :: label
        complex(pReal), intent(in) :: tensor(:, :, :, :, :, :, :)
        print *, label, shape(tensor), "vals:", tensor
    end subroutine p7c

end module tensor_printer

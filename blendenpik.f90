! Blendenpik sketch-and-precondition algorithm.
program blendenpik

end program blendenpik


module blendenpik_module
  implicit none (type, external)

  include 'fftw/api/fftw3.f'

  external dgeqrf, dgels, dtrtrs, dfftw_plan_r2r_1d, dfftw_execute_r2r, dfftw_destroy_plan

  private
  public :: blendenpik

contains
  subroutine blendenpik(A, b, x, m, n, s)
    ! Solves the linear system Ax = b using the Blendenpik algorithm.
    ! A: Input matrix (m x n)
    ! b: Right-hand side vector (m)
    ! x: Solution vector (n)
    ! s: Sketch size

    real(8), intent(in) :: A(:,:), b(:)
    real(8), intent(out) :: x(:)
    integer, intent(in) :: s, m, n

    integer :: i, info, lwork, plan

    real(8) :: rand_no
    integer, dimension(s) :: row_indices
    real(8), dimension(s) :: harvest
    real(8) :: work_query(1)
    real(8), dimension(m, n) :: DCT_M, tau
    real(8), dimension(s, n) :: SM
    real(8), dimension(n, n) :: R
    real(8), allocatable :: work(:)


    ! create DA, which is A with random sign flips
    ! i.e. D is a random diagonal matrix with +/- 1 on the diagonal
    call random_seed()
    do i = 1, m
      call random_number(rand_no)
      if (rand_no < 0.5d0) then
        DCT_M(i, :) = -A(i, :)
      else
        DCT_M(i, :) = A(i, :)
      end if
    end do

    ! apply discrete cosine transform to each column of DA using fftw
    call dfftw_plan_r2r_1d(plan, m, DCT_M(:, 1), DCT_M(:, 1), FFTW_REDFT10, FFTW_ESTIMATE)
    do i = 1, n
      call dfftw_execute_r2r(plan, DCT_M(:, i), DCT_M(:, i))
    end do 
    call dfftw_destroy_plan(plan)

    ! sketch by selecting s random rows of M
    call random_number(harvest)
    row_indices = mod(int(row_indices * m), m) + 1
    do i = 1, s
      SM(i, :) = DCT_M(row_indices(i), :)
    end do

    ! now take QR decomposition of SM
    ! workspace query
    call dgeqrf(s, n, SM, s, tau, work_query, -1, info)
    lwork = int(work_query(1))
    allocate(work(lwork))
    call dgeqrf(s, n, SM, s, tau, work, lwork, info)
    deallocate(work)

    ! extract R from QR decomposition
    R = 0.0d0
    do i = 1, n
      R(i, i:n) = SM(i, i:n)
    end do

    ! now solve the preconditioned least squares problem
    ! i.e. minimize ||A * R^{-1} y - b||_2
    ! then set x = R^{-1} y
    ! workspace query
    call dgels('N', m, n, 1, A, m, b, m, x, n, work_query, -1, info)
    lwork = int(work_query(1))
    allocate(work(lwork))
    call dgels('N', m, n, 1, A, m, b, m, x, n, work, lwork, info)
    call dtrtrs('U', 'N', 'N', n, 1, R, n, x, n, info)
    deallocate(work)

  end subroutine blendenpik
end module blendenpik_module

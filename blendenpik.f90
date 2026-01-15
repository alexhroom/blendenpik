! Blendenpik sketch-and-precondition algorithm.

module blendenpik_module
  implicit none (external)
  private
  public :: blendenpik

contains
  subroutine blendenpik(A, b, x, s)
    ! Solves the linear system Ax = b using the Blendenpik algorithm.
    ! A: Input matrix (m x n)
    ! b: Right-hand side vector (m)
    ! x: Solution vector (n)
    ! s: Sketch size

    real(real64), intent(in) :: A(:,:), b(:)
    real(real64), intent(out) :: x(:)
    integer, intent(in) :: s

    integer :: m, n, i, info, lwork
    integer, dimension(s) :: row_indices
    real(real64), dimension(m, n) :: M
    real(real64), dimension(m, m) :: D
    real(real64), dimension(s, n) :: SM
    real(real64) :: R(n, n)

    m = size(A, 1)
    n = size(A, 2)

    ! create diagonal matrix D with random ±1 entries
    real(real64) :: D(:)
    real(real64) :: diag(n)
    D = 0.0d0
    where (diag > 0.5d0)
      diag = 1.0d0
    elsewhere
      diag = -1.0d0
    end where
    do i = 1, m
      D(i, i) = diag(i)
    end do

    ! compute M which is the DCT of DA
    call dgemm('N', 'N', m, n, n, 1.0d0, D, m, A, n, 0.0d0, M, m)

    !call dct(......)

    ! sketch by selecting s random rows of M
    call random_seed()
    call random_number(row_indices)
    row_indices = mod(int(row_indices * m), m) + 1
    do i = 1, s
      SM(i, :) = M(row_indices(i), :)
    end do

    ! now take QR decomposition of SM
    ! do workspace stuff...
    call dgeqrf(s, n, SM, s, tau, work, lwork, info)

    ! extract R from QR decomposition
    R = 0.0d0
    do i = 1, n
      R(i, i:n) = SM(i, i:n)
    end do

    ! then do LSQR to solve the preconditioned system
    ! and then recover x
  end subroutine blendenpik
  end module blendenpik_module

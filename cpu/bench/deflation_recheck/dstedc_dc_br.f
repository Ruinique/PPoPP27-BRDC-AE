*> \brief \b DSTEDC_DC_BR private values-only DC boundary-row driver.
*
*  DSTEDC_DC_BR is an experimental values-only entry point for local
*  testing of DLAED0_BR.  It intentionally does not change DSTEDC('N')
*  or any public LAPACK/LAPACKE interface.
*
      SUBROUTINE DSTEDC_DC_BR( N, D, E, WORK, LWORK, IWORK, LIWORK,
     $                         INFO )
*
*     .. Scalar Arguments ..
      INTEGER            INFO, LIWORK, LWORK, N
*     ..
*     .. Array Arguments ..
      INTEGER            IWORK( * )
      DOUBLE PRECISION   D( * ), E( * ), WORK( * )
*     ..
*
*  =====================================================================
*
*     .. Parameters ..
      DOUBLE PRECISION   ZERO, ONE
      PARAMETER          ( ZERO = 0.0D0, ONE = 1.0D0 )
*     ..
*     .. Local Scalars ..
      LOGICAL            LQUERY
      INTEGER            FINISH, LIWMIN, LWMIN, M, SMLSIZ, START
      DOUBLE PRECISION   EPS, ORGNRM, TINY
*     ..
*     .. External Functions ..
      INTEGER            ILAENV
      DOUBLE PRECISION   DLAMCH, DLANST
      EXTERNAL           ILAENV, DLAMCH, DLANST
*     ..
*     .. External Subroutines ..
      EXTERNAL           DLAED0_BR, DLASCL, DLASRT, DSTERF, XERBLA
*     ..
*     .. Intrinsic Functions ..
      INTRINSIC          ABS, MAX, MOD, SQRT
*     ..
*     .. Executable Statements ..
*
      INFO = 0
      LQUERY = ( LWORK.EQ.-1 .OR. LIWORK.EQ.-1 )
*
      IF( N.LT.0 ) THEN
         INFO = -1
      END IF
*
      IF( INFO.EQ.0 ) THEN
         SMLSIZ = ILAENV( 9, 'DSTEDC', ' ', 0, 0, 0, 0 )
         IF( N.LE.SMLSIZ ) THEN
            LWMIN = 1
            LIWMIN = 1
         ELSE
            LWMIN = MAX( 1, 16*N )
            LIWMIN = MAX( 1, 7*N )
         END IF
         WORK( 1 ) = LWMIN
         IWORK( 1 ) = LIWMIN
*
         IF( LWORK.LT.LWMIN .AND. .NOT.LQUERY ) THEN
            INFO = -5
         ELSE IF( LIWORK.LT.LIWMIN .AND. .NOT.LQUERY ) THEN
            INFO = -7
         END IF
      END IF
*
      IF( INFO.NE.0 ) THEN
         CALL XERBLA( 'DSTEDC_DC_BR', -INFO )
         RETURN
      ELSE IF( LQUERY ) THEN
         RETURN
      END IF
*
      IF( N.EQ.0 )
     $   RETURN
      IF( N.EQ.1 )
     $   RETURN
*
      IF( N.LE.SMLSIZ ) THEN
         CALL DSTERF( N, D, E, INFO )
         GO TO 50
      END IF
*
      ORGNRM = DLANST( 'M', N, D, E )
      IF( ORGNRM.LE.ZERO )
     $   GO TO 50
*
      EPS = DLAMCH( 'Epsilon' )
      START = 1
*
   10 CONTINUE
      IF( START.LE.N ) THEN
         FINISH = START
   20    CONTINUE
         IF( FINISH.LT.N ) THEN
            TINY = EPS*SQRT( ABS( D( FINISH ) ) )*
     $             SQRT( ABS( D( FINISH+1 ) ) )
            IF( ABS( E( FINISH ) ).GT.TINY ) THEN
               FINISH = FINISH + 1
               GO TO 20
            END IF
         END IF
*
         M = FINISH - START + 1
         IF( M.EQ.1 ) THEN
            START = FINISH + 1
            GO TO 10
         END IF
*
         IF( M.GT.SMLSIZ ) THEN
            ORGNRM = DLANST( 'M', M, D( START ), E( START ) )
            CALL DLASCL( 'G', 0, 0, ORGNRM, ONE, M, 1, D( START ),
     $                   M, INFO )
            CALL DLASCL( 'G', 0, 0, ORGNRM, ONE, M-1, 1,
     $                   E( START ), M-1, INFO )
*
            CALL DLAED0_BR( M, D( START ), E( START ), WORK, IWORK,
     $                      INFO )
            IF( INFO.NE.0 ) THEN
               INFO = ( INFO / ( M+1 )+START-1 )*( N+1 ) +
     $                MOD( INFO, ( M+1 ) ) + START - 1
               GO TO 50
            END IF
*
            CALL DLASCL( 'G', 0, 0, ONE, ORGNRM, M, 1, D( START ),
     $                   M, INFO )
         ELSE
            CALL DSTERF( M, D( START ), E( START ), INFO )
            IF( INFO.NE.0 ) THEN
               INFO = START*( N+1 ) + FINISH
               GO TO 50
            END IF
         END IF
*
         START = FINISH + 1
         GO TO 10
      END IF
*
      CALL DLASRT( 'I', N, D, INFO )
*
   50 CONTINUE
      WORK( 1 ) = LWMIN
      IWORK( 1 ) = LIWMIN
      RETURN
*
*     End of DSTEDC_DC_BR
*
      END

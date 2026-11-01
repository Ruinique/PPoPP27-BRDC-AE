*> \brief \b DLAED0_BR values-only DC prototype with boundary rows.
*
*  DLAED0_BR computes eigenvalues of a symmetric tridiagonal matrix with
*  a private divide-and-conquer prototype.  Unlike DLAED0(ICOMPQ=0), it
*  keeps only first/last boundary rows for each active block across
*  merge levels instead of saving complete eigenvector blocks.
*
      SUBROUTINE DLAED0_BR( N, D, E, WORK, IWORK, INFO )
*
*     .. Scalar Arguments ..
      INTEGER            INFO, N
*     ..
*     .. Array Arguments ..
      INTEGER            IWORK( * )
      DOUBLE PRECISION   D( * ), E( * ), WORK( * )
*     ..
*
*  =====================================================================
*
*     .. Parameters ..
      DOUBLE PRECISION   TAU_LOW, ZERO
      INTEGER            INNER_CAP
      PARAMETER          ( INNER_CAP = 8, TAU_LOW = 0.20D0,
     $                     ZERO = 0.0D0 )
*     ..
*     .. Local Scalars ..
      LOGICAL            OUTER_ONLY, USE_LEVEL_PAR, WANTQ
      INTEGER            ERRSIZ, ERRSUB, I, IBHI, IBLO, IIWRK, INDXQ,
     $                   INFOM, ISCR, IWBASE, J, K, KDEFL, LEAFQ,
     $                   LEAF_THREADS, LEFT, MATSIZ, MAX_THREADS,
     $                   MERGE, MERGE_LO, MSD2, NMERGE, NTHR,
     $                   OUTER_THREADS, QBASE, RIGHT, SMLSIZ, SMM1,
     $                   SPM1, SUBMAT, SUBPBS, TID, TLVLS, WBASE,
     $                   WRKBASE, WRKSTR
      DOUBLE PRECISION   RHO
      CHARACTER*32       SCHED_ENV
*     ..
*     .. External Subroutines ..
      EXTERNAL           BRDC_WPROD_NTHREADS, DCOPY, DLAED7_BR,
     $                   DSTEQR_BOUNDARY_BR, XERBLA
*     ..
*     .. External Functions ..
      INTEGER            BRDC_WPROD_NTHREADS, ILAENV,
     $                   OMP_GET_MAX_THREADS, OMP_GET_THREAD_NUM
      EXTERNAL           ILAENV, OMP_GET_MAX_THREADS, OMP_GET_THREAD_NUM
*     ..
*     .. Intrinsic Functions ..
      INTRINSIC          ABS, DBLE, MAX, MIN
*     ..
*     .. Executable Statements ..
*
      INFO = 0
      IF( N.LT.0 ) THEN
         INFO = -1
      END IF
      IF( INFO.NE.0 ) THEN
         CALL XERBLA( 'DLAED0_BR', -INFO )
         RETURN
      END IF
*
      IF( N.EQ.0 )
     $   RETURN
*
      SMLSIZ = ILAENV( 9, 'DLAED0', ' ', 0, 0, 0, 0 )
      MAX_THREADS = 1
!$    MAX_THREADS = OMP_GET_MAX_THREADS()
      OUTER_ONLY = .FALSE.
      SCHED_ENV = ' '
      CALL GET_ENVIRONMENT_VARIABLE( 'OPENBLAS_BRDC_SCHED',
     $                               SCHED_ENV )
      IF( SCHED_ENV( 1:1 ).EQ.'o' .OR.
     $    SCHED_ENV( 1:1 ).EQ.'O' ) THEN
         OUTER_ONLY = .TRUE.
      END IF
*
*     Determine the size and placement of leaf submatrices.
*
      IWORK( 1 ) = N
      SUBPBS = 1
      TLVLS = 0
   10 CONTINUE
      IF( IWORK( SUBPBS ).GT.SMLSIZ ) THEN
         DO 20 J = SUBPBS, 1, -1
            IWORK( 2*J ) = ( IWORK( J )+1 ) / 2
            IWORK( 2*J-1 ) = IWORK( J ) / 2
   20    CONTINUE
         TLVLS = TLVLS + 1
         SUBPBS = 2*SUBPBS
         GO TO 10
      END IF
      DO 30 J = 2, SUBPBS
         IWORK( J ) = IWORK( J ) + IWORK( J-1 )
   30 CONTINUE
*
*     Apply rank-1 cuts.
*
      SPM1 = SUBPBS - 1
      DO 40 I = 1, SPM1
         SUBMAT = IWORK( I ) + 1
         SMM1 = SUBMAT - 1
         D( SMM1 ) = D( SMM1 ) - ABS( E( SMM1 ) )
         D( SUBMAT ) = D( SUBMAT ) - ABS( E( SMM1 ) )
   40 CONTINUE
*
*     Workspace layout:
*       WORK(IBLO:IBLO+N-1)       first boundary rows
*       WORK(IBHI:IBHI+N-1)       last boundary rows
*       WORK(ISCR:)               leaf scratch and per-merge scratch slices
*
*     Integer workspace layout:
*       IWORK(1:N)                split endpoints
*       IWORK(INDXQ:INDXQ+N-1)   sorting permutations
*       IWORK(IIWRK:)             DLAED7_BR scratch
*
      IBLO = 1
      IBHI = IBLO + N
      ISCR = IBHI + N
      INDXQ = N + 1
      IIWRK = INDXQ + N
      NTHR = BRDC_WPROD_NTHREADS()
      WRKSTR = 14 + 2*NTHR
*
*     Solve leaf eigenproblems and initialize boundary row invariants.
*     Leaves are independent.  Reuse the public WORK scratch with one
*     DSTEQR-sized slice per OpenMP thread instead of per-leaf stack Q.
*
      LEAFQ = SMLSIZ*SMLSIZ + MAX( 1, 2*SMLSIZ )
      LEAF_THREADS = MAX( 1, MIN( MAX_THREADS, WRKSTR*N / LEAFQ ) )
      ERRSUB = 0
      ERRSIZ = 0
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(I,SUBMAT,MATSIZ,J,K,INFOM,
!$OMP& TID,QBASE,WBASE)
!$OMP& IF(SUBPBS.GT.4 .AND. LEAF_THREADS.GT.1)
!$OMP& NUM_THREADS(LEAF_THREADS) SCHEDULE(STATIC)
      DO 70 I = 1, SUBPBS
         INFOM = 0
         TID = 0
!$       TID = OMP_GET_THREAD_NUM()
         QBASE = ISCR + TID*LEAFQ
         WBASE = QBASE + SMLSIZ*SMLSIZ
         IF( I.EQ.1 ) THEN
            SUBMAT = 1
            MATSIZ = IWORK( 1 )
         ELSE
            SUBMAT = IWORK( I-1 ) + 1
            MATSIZ = IWORK( I ) - IWORK( I-1 )
         END IF
*
         CALL DSTEQR_BOUNDARY_BR( MATSIZ, D( SUBMAT ), E( SUBMAT ),
     $                            WORK( IBLO+SUBMAT-1 ),
     $                            WORK( IBHI+SUBMAT-1 ),
     $                            WORK( QBASE ), SMLSIZ, WORK( WBASE ),
     $                            INFOM )
         IF( INFOM.NE.0 ) THEN
!$OMP CRITICAL
            IF( INFO.EQ.0 ) THEN
               INFO = INFOM
               ERRSUB = SUBMAT
               ERRSIZ = MATSIZ
            END IF
!$OMP END CRITICAL
         ELSE
            K = 1
            DO 65 J = SUBMAT, IWORK( I )
               IWORK( INDXQ+J-1 ) = K
               K = K + 1
   65       CONTINUE
         END IF
   70 CONTINUE
!$OMP END PARALLEL DO
      IF( INFO.NE.0 ) THEN
         SUBMAT = ERRSUB
         MATSIZ = ERRSIZ
         GO TO 130
      END IF
*
*     Merge adjacent eigensystems.  Boundary rows are updated in place
*     for the merged physical D order.
*
   80 CONTINUE
      IF( SUBPBS.GT.1 ) THEN
         WANTQ = SUBPBS.GT.2
         ERRSUB = 0
         ERRSIZ = 0
         NMERGE = SUBPBS / 2
         USE_LEVEL_PAR = .FALSE.
         OUTER_THREADS = 1
         MERGE_LO = 1
*
*        When a level has enough independent merge pairs to occupy the
*        whole thread budget, run it outer-parallel without a serial
*        probe.  This keeps pseudo-random spectra on the old fast path
*        where early levels have many cheap, strongly deflating merges.
*        Once the tree narrows, probe the first merge at this level and
*        choose scheduling from rho = KDEFL/MATSIZ:
*          rho <= TAU_LOW  -> outer level-parallel (structured, small K)
*          otherwise       -> conservative inner-first
*
         IF( OUTER_ONLY .AND. NMERGE.GT.1 .AND.
     $       MAX_THREADS.GT.1 ) THEN
            USE_LEVEL_PAR = .TRUE.
            OUTER_THREADS = MIN( NMERGE, MAX_THREADS )
         ELSE IF( 4*NMERGE.GE.MAX_THREADS .AND.
     $            MAX_THREADS.GT.1 ) THEN
            USE_LEVEL_PAR = .TRUE.
            OUTER_THREADS = MIN( NMERGE, MAX_THREADS )
         ELSE IF( NMERGE.GT.1 .AND. MAX_THREADS.GT.INNER_CAP ) THEN
            MERGE = 1
            LEFT = 1
            RIGHT = 2
            SUBMAT = 1
            MATSIZ = IWORK( RIGHT )
            MSD2 = IWORK( 1 )
            WRKBASE = ISCR + WRKSTR*( SUBMAT-1 )
            IWBASE = IIWRK + 5*( SUBMAT-1 )
            INFOM = 0
            CALL DLAED7_BR( WANTQ, MATSIZ, MSD2, D( SUBMAT ),
     $                      IWORK( INDXQ+SUBMAT-1 ),
     $                      E( SUBMAT+MSD2-1 ),
     $                      WORK( IBLO+SUBMAT-1 ),
     $                      WORK( IBHI+SUBMAT-1 ), WORK( WRKBASE ),
     $                      IWORK( IWBASE ), KDEFL, INFOM )
            IF( INFOM.NE.0 ) THEN
               INFO = INFOM
               ERRSUB = SUBMAT
               ERRSIZ = MATSIZ
               GO TO 130
            END IF
            IF( MATSIZ.GT.0 ) THEN
               RHO = DBLE( KDEFL ) / DBLE( MATSIZ )
            ELSE
               RHO = ZERO
            END IF
            IF( RHO.LE.TAU_LOW .AND. NMERGE.GT.2 ) THEN
               USE_LEVEL_PAR = .TRUE.
            END IF
            IF( USE_LEVEL_PAR ) THEN
               OUTER_THREADS = MIN( NMERGE-1, MAX_THREADS )
               OUTER_THREADS = MAX( 1, OUTER_THREADS )
            END IF
            MERGE_LO = 2
         END IF
*
         IF( MERGE_LO.LE.NMERGE ) THEN
            IF( USE_LEVEL_PAR ) THEN
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(MERGE,LEFT,RIGHT,SUBMAT,
!$OMP& MATSIZ,MSD2,WRKBASE,IWBASE,INFOM,KDEFL)
!$OMP& NUM_THREADS(OUTER_THREADS) SCHEDULE(STATIC)
               DO 90 MERGE = MERGE_LO, NMERGE
                  CALL DLAED0_MERGE_PAIR( MERGE, NMERGE, IWORK, WANTQ,
     $                 D, E, WORK, IBLO, IBHI, ISCR, IIWRK, INDXQ,
     $                 SUBMAT, MATSIZ, MSD2, WRKBASE, IWBASE, KDEFL,
     $                 INFOM, WRKSTR )
                  IF( INFOM.NE.0 ) THEN
!$OMP CRITICAL
                     IF( INFO.EQ.0 ) THEN
                        INFO = INFOM
                        ERRSUB = SUBMAT
                        ERRSIZ = MATSIZ
                     END IF
!$OMP END CRITICAL
                  END IF
   90          CONTINUE
!$OMP END PARALLEL DO
            ELSE
               DO 91 MERGE = MERGE_LO, NMERGE
                  CALL DLAED0_MERGE_PAIR( MERGE, NMERGE, IWORK, WANTQ,
     $                 D, E, WORK, IBLO, IBHI, ISCR, IIWRK, INDXQ,
     $                 SUBMAT, MATSIZ, MSD2, WRKBASE, IWBASE, KDEFL,
     $                 INFOM, WRKSTR )
                  IF( INFOM.NE.0 ) THEN
                     INFO = INFOM
                     ERRSUB = SUBMAT
                     ERRSIZ = MATSIZ
                     GO TO 130
                  END IF
   91          CONTINUE
            END IF
         END IF
         IF( INFO.NE.0 ) THEN
            SUBMAT = ERRSUB
            MATSIZ = ERRSIZ
            GO TO 130
         END IF
*
         DO 95 MERGE = 1, NMERGE
            IWORK( MERGE ) = IWORK( 2*MERGE )
   95    CONTINUE
         SUBPBS = SUBPBS / 2
         GO TO 80
      END IF
*
*     Reorder eigenvalues from the final physical merge order.
*
      DO 100 I = 1, N
         J = IWORK( INDXQ+I-1 )
         WORK( ISCR+I-1 ) = D( J )
  100 CONTINUE
      CALL DCOPY( N, WORK( ISCR ), 1, D, 1 )
      GO TO 140
*
  130 CONTINUE
      INFO = SUBMAT*( N+1 ) + SUBMAT + MATSIZ - 1
*
  140 CONTINUE
      RETURN
*
*     End of DLAED0_BR
*
      END
*
*> \brief \b DLAED0_MERGE_PAIR executes one merge in the DC tree.
*
      SUBROUTINE DLAED0_MERGE_PAIR( MERGE, NMERGE, IWORK, WANTQ, D, E,
     $                               WORK, IBLO, IBHI, ISCR, IIWRK,
     $                               INDXQ, SUBMAT, MATSIZ, MSD2,
     $                               WRKBASE, IWBASE, KDEFL, INFO,
     $                               WRKSTR )
*
      INTEGER            IBHI, IBLO, IIWRK, INDXQ, INFO, ISCR, IWBASE,
     $                   KDEFL, MATSIZ, MERGE, MSD2, NMERGE, RIGHT,
     $                   SUBMAT, WRKBASE, WRKSTR
      LOGICAL            WANTQ
      INTEGER            IWORK( * )
      DOUBLE PRECISION   D( * ), E( * ), WORK( * )
      EXTERNAL           DLAED7_BR
*
      INTEGER            LEFT
*
      LEFT = 2*MERGE - 1
      RIGHT = LEFT + 1
      IF( MERGE.EQ.1 ) THEN
         SUBMAT = 1
         MATSIZ = IWORK( RIGHT )
         MSD2 = IWORK( 1 )
      ELSE
         SUBMAT = IWORK( LEFT-1 ) + 1
         MATSIZ = IWORK( RIGHT ) - IWORK( LEFT-1 )
         MSD2 = IWORK( LEFT ) - IWORK( LEFT-1 )
      END IF
*
      WRKBASE = ISCR + WRKSTR*( SUBMAT-1 )
      IWBASE = IIWRK + 5*( SUBMAT-1 )
*
      CALL DLAED7_BR( WANTQ, MATSIZ, MSD2, D( SUBMAT ),
     $                IWORK( INDXQ+SUBMAT-1 ), E( SUBMAT+MSD2-1 ),
     $                WORK( IBLO+SUBMAT-1 ), WORK( IBHI+SUBMAT-1 ),
     $                WORK( WRKBASE ), IWORK( IWBASE ), KDEFL, INFO )
      RETURN
      END
*
*> \brief \b DSTEQR_BOUNDARY_BR solves one leaf and keeps boundary rows.
*
      SUBROUTINE DSTEQR_BOUNDARY_BR( N, D, E, BLO, BHI, Q, LDQ, WORK,
     $                               INFO )
*
      INTEGER            INFO, J, LDQ, N
      DOUBLE PRECISION   BLO( * ), BHI( * ), D( * ), E( * ), Q( * ),
     $                   WORK( * )
      EXTERNAL           DSTEQR
*
      CALL DSTEQR( 'I', N, D, E, Q, LDQ, WORK, INFO )
      IF( INFO.NE.0 )
     $   RETURN
*
      DO 10 J = 1, N
         BLO( J ) = Q( 1 + ( J - 1 )*LDQ )
         BHI( J ) = Q( N + ( J - 1 )*LDQ )
   10 CONTINUE
      RETURN
      END

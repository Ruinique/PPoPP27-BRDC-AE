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
      DOUBLE PRECISION   TAU_HIGH, TAU_LOW, ZERO
      INTEGER            INNER_CAP, MERGE_RWORK
      PARAMETER          ( INNER_CAP = 8, MERGE_RWORK = 24,
     $                     TAU_HIGH = 0.40D0, TAU_LOW = 0.20D0,
     $                     ZERO = 0.0D0 )
*     ..
*     .. Local Scalars ..
      LOGICAL            USE_LEVEL_PAR, WANTQ
      INTEGER            ERRSIZ, ERRSUB, I, IBHI, IBLO, IIWRK, INDXQ,
     $                   INFOM, ISCR, IWBASE, J, K, KDEFL, LEAFQ, LEFT,
     $                   MATSIZ, MAX_THREADS, MERGE,
     $                   MERGE_LO, MSD2, NMERGE, OUTER_THREADS, QBASE,
     $                   RIGHT, SMLSIZ, SMM1, SPM1, SUBMAT, SUBPBS,
     $                   NUMA_DOMAINS, NUM_PROCS, SOCKET_THREADS,
     $                   TEAM_BUDGET, TID, TLVLS, WBASE, WRKBASE
      DOUBLE PRECISION   RHO
*     ..
*     .. External Subroutines ..
      EXTERNAL           DCOPY, DLAED7_BR, DSTEQR_BOUNDARY_BR, XERBLA
*     ..
*     .. External Functions ..
      INTEGER            ILAENV, OMP_GET_MAX_THREADS, OMP_GET_NUM_PROCS,
     $                   OMP_GET_THREAD_NUM, OPENBLAS_GET_NUM_NODES
      EXTERNAL           ILAENV, OMP_GET_MAX_THREADS, OMP_GET_NUM_PROCS,
     $                   OMP_GET_THREAD_NUM, OPENBLAS_GET_NUM_NODES
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
      NUM_PROCS = MAX_THREADS
!$    NUM_PROCS = OMP_GET_NUM_PROCS()
      NUMA_DOMAINS = 1
      NUMA_DOMAINS = OPENBLAS_GET_NUM_NODES()
      NUMA_DOMAINS = MAX( 1, MIN( NUMA_DOMAINS, NUM_PROCS ) )
      SOCKET_THREADS = MAX_THREADS
      IF( NUM_PROCS.GT.NUMA_DOMAINS*INNER_CAP )
     $   SOCKET_THREADS = MIN( MAX_THREADS,
     $      ( NUM_PROCS+NUMA_DOMAINS-1 ) / NUMA_DOMAINS )
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
*
*     Solve leaf eigenproblems and initialize boundary row invariants.
*     Leaves are independent.  Reuse the public WORK scratch with one
*     DSTEQR-sized slice per OpenMP thread instead of per-leaf stack Q.
*
      LEAFQ = SMLSIZ*SMLSIZ + MAX( 1, 2*SMLSIZ )
      ERRSUB = 0
      ERRSIZ = 0
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(I,SUBMAT,MATSIZ,J,K,INFOM,
!$OMP& TID,QBASE,WBASE)
!$OMP& IF(SUBPBS.GT.4) SCHEDULE(STATIC)
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
*        Use a NUMA-local thread budget once the machine is wider than a
*        useful inner team.  The OpenBLAS NUMA helper reports configured
*        domains; a single-domain system naturally keeps the full thread
*        budget.  This avoids crossing sockets on narrow levels unless
*        there are enough independent merge pairs to keep the full machine
*        busy.
*
*        If a level has enough merge pairs to occupy the chosen budget, run
*        it outer-parallel without a serial probe.  Otherwise probe the
*        first merge at this level and choose scheduling from
*        rho = KDEFL/MATSIZ:
*          rho <= TAU_LOW  -> outer level-parallel (structured, small K)
*          rho >= TAU_HIGH -> outer level-parallel on the NUMA-local
*                             budget
*          otherwise       -> conservative inner-first
*
         TEAM_BUDGET = MAX_THREADS
         IF( SOCKET_THREADS.LT.MAX_THREADS )
     $      TEAM_BUDGET = SOCKET_THREADS
         IF( NMERGE.GE.MAX( 1, TEAM_BUDGET ) ) THEN
            USE_LEVEL_PAR = .TRUE.
            OUTER_THREADS = TEAM_BUDGET
            IF( MAX_THREADS.GT.SOCKET_THREADS .AND.
     $          NMERGE.GE.MAX_THREADS )
     $         OUTER_THREADS = MAX_THREADS
         ELSE IF( NMERGE.GT.1 .AND. MAX_THREADS.GT.INNER_CAP ) THEN
            MERGE = 1
            LEFT = 1
            RIGHT = 2
            SUBMAT = 1
            MATSIZ = IWORK( RIGHT )
            MSD2 = IWORK( 1 )
            WRKBASE = ISCR + MERGE_RWORK*( SUBMAT-1 )
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
            ELSE IF( RHO.GE.TAU_HIGH .AND.
     $               TEAM_BUDGET.GT.INNER_CAP ) THEN
               USE_LEVEL_PAR = .TRUE.
            END IF
            IF( USE_LEVEL_PAR ) THEN
               OUTER_THREADS = MIN( NMERGE-1, TEAM_BUDGET )
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
     $                 INFOM )
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
     $                 INFOM )
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
     $                               WRKBASE, IWBASE, KDEFL, INFO )
*
      INTEGER            IBHI, IBLO, IIWRK, INDXQ, INFO, ISCR, IWBASE,
     $                   KDEFL, MATSIZ, MERGE, MSD2, NMERGE, RIGHT,
     $                   SUBMAT, WRKBASE
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
      WRKBASE = ISCR + 24*( SUBMAT-1 )
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

*> \brief \b DLAED9_BR selected-row secular update helper.
*
*  DLAED9_BR finds the roots of the secular equation for one
*  boundary-row DC merge and directly updates a small set of selected
*  rows.  It mirrors DLAED9's root solve and eigenvector normalization,
*  but avoids materializing the full K-by-K secular eigenvector block.
*  DLAED4_BR returns high-relative-accuracy DELTA values for each root;
*  consume those DELTA values immediately to form WPROD, following the
*  same numerical idea as LAPACK's SVD divide-and-conquer helpers.
*
      SUBROUTINE DLAED9_BR( K, NBR, D, QIN, LDQIN, QOUT, LDQOUT,
     $                       RHO, DLAMBDA, W, DELTA, WPROD, WSGN,
     $                       TAU, NEAR, WACC, DWORK, INFO )
*
*     .. Scalar Arguments ..
      INTEGER            INFO, K, LDQIN, LDQOUT, NBR
      DOUBLE PRECISION   RHO
*     ..
*     .. Array Arguments ..
      DOUBLE PRECISION   D( * ), DELTA( * ), DLAMBDA( * ),
     $                   DWORK( * ),
     $                   NEAR( 2, * ), QIN( LDQIN, * ),
     $                   QOUT( LDQOUT, * ), TAU( * ), W( * ),
     $                   WACC( * ), WPROD( * ), WSGN( * )
*     ..
*
*  =====================================================================
*
*     .. Parameters ..
      DOUBLE PRECISION   ZERO, ONE
      PARAMETER          ( ZERO = 0.0D0, ONE = 1.0D0 )
*     ..
*     .. Local Scalars ..
      INTEGER            I, INFOJ, IOFF, IORG, J, N1, N2, NTHREADS,
     $                   R, TID
      DOUBLE PRECISION   ACC1, ACC2, DLT, ROWSUM, TEMP
*     ..
*     .. External Functions ..
      INTEGER            BRDC_WPROD_NTHREADS, OMP_GET_MAX_THREADS,
     $                   OMP_GET_THREAD_NUM
      EXTERNAL           BRDC_WPROD_NTHREADS, OMP_GET_MAX_THREADS,
     $                   OMP_GET_THREAD_NUM
*     ..
*     .. External Subroutines ..
      EXTERNAL           DLAED4, DLAED4_BR, XERBLA
*     ..
*     .. Intrinsic Functions ..
      INTRINSIC          ABS, MAX, MIN, MOD, SIGN, SQRT
*     ..
*     .. Executable Statements ..
*
      INFO = 0
      IF( K.LT.0 ) THEN
         INFO = -1
      ELSE IF( NBR.LT.0 ) THEN
         INFO = -2
      ELSE IF( LDQIN.LT.MAX( 1, NBR ) ) THEN
         INFO = -5
      ELSE IF( LDQOUT.LT.MAX( 1, NBR ) ) THEN
         INFO = -7
      END IF
      IF( INFO.NE.0 ) THEN
         CALL XERBLA( 'DLAED9_BR', -INFO )
         RETURN
      END IF
*
      IF( K.EQ.0 )
     $   RETURN
*
*     Root-only final merge: no parent boundary rows will be observed by
*     a higher merge level, so updated W and secular vectors are not
*     needed.  Reuse otherwise idle O(K) scratch slices as per-thread
*     DLAED4 delta buffers; this keeps the public O(N) workspace bound.
*     In this path QIN and QOUT are two unused 2-by-K work arrays, so they
*     provide four more delta buffers without changing the workspace query.
*
      IF( NBR.EQ.0 ) THEN
         NTHREADS = 1
!$       NTHREADS = MIN( 10, OMP_GET_MAX_THREADS() )
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(J,TID,INFOJ)
!$OMP& IF(K.GT.128) NUM_THREADS(NTHREADS) SCHEDULE(STATIC)
         DO 5 J = 1, K
            TID = 0
            INFOJ = 0
!$          TID = OMP_GET_THREAD_NUM()
            IF( TID.EQ.0 ) THEN
               CALL DLAED4( K, J, DLAMBDA, W, DELTA, RHO, D( J ),
     $                      INFOJ )
            ELSE IF( TID.EQ.1 ) THEN
               CALL DLAED4( K, J, DLAMBDA, W, WPROD, RHO, D( J ),
     $                      INFOJ )
            ELSE IF( TID.EQ.2 ) THEN
               CALL DLAED4( K, J, DLAMBDA, W, WPROD( K+1 ), RHO,
     $                      D( J ), INFOJ )
            ELSE IF( TID.EQ.3 ) THEN
               CALL DLAED4( K, J, DLAMBDA, W, WSGN, RHO, D( J ),
     $                      INFOJ )
            ELSE IF( TID.EQ.4 ) THEN
               CALL DLAED4( K, J, DLAMBDA, W, TAU, RHO, D( J ),
     $                      INFOJ )
            ELSE IF( TID.EQ.5 ) THEN
               CALL DLAED4( K, J, DLAMBDA, W, NEAR, RHO, D( J ),
     $                      INFOJ )
            ELSE IF( TID.EQ.6 ) THEN
               CALL DLAED4( K, J, DLAMBDA, W, QIN, RHO, D( J ),
     $                      INFOJ )
            ELSE IF( TID.EQ.7 ) THEN
               IF( MOD( K, 2 ).EQ.0 ) THEN
                  CALL DLAED4( K, J, DLAMBDA, W, QIN( 1, K/2+1 ),
     $                         RHO, D( J ), INFOJ )
               ELSE
                  CALL DLAED4( K, J, DLAMBDA, W,
     $                         QIN( 2, ( K+1 )/2 ), RHO, D( J ),
     $                         INFOJ )
               END IF
            ELSE IF( TID.EQ.8 ) THEN
               CALL DLAED4( K, J, DLAMBDA, W, QOUT, RHO, D( J ),
     $                      INFOJ )
            ELSE
               IF( MOD( K, 2 ).EQ.0 ) THEN
                  CALL DLAED4( K, J, DLAMBDA, W, QOUT( 1, K/2+1 ),
     $                         RHO, D( J ), INFOJ )
               ELSE
                  CALL DLAED4( K, J, DLAMBDA, W,
     $                         QOUT( 2, ( K+1 )/2 ), RHO, D( J ),
     $                         INFOJ )
               END IF
            END IF
            IF( INFOJ.NE.0 ) THEN
!$OMP CRITICAL
               IF( INFO.EQ.0 )
     $            INFO = INFOJ
!$OMP END CRITICAL
            END IF
    5    CONTINUE
!$OMP END PARALLEL DO
         IF( INFO.NE.0 )
     $      GO TO 170
         GO TO 170
      END IF
*
*     For K = 1 or 2, DLAED4/DLAED5 returns normalized eigenvectors
*     directly in DELTA.  Keep this path identical to DLAED9's
*     special-case copy to S.
*
      IF( K.EQ.1 .OR. K.EQ.2 ) THEN
         DO 30 J = 1, K
            CALL DLAED4( K, J, DLAMBDA, W, DELTA, RHO, D( J ),
     $                   INFO )
            IF( INFO.NE.0 )
     $         GO TO 170
            IF( NBR.EQ.2 ) THEN
               ACC1 = ZERO
               ACC2 = ZERO
               DO 10 I = 1, K
                  ACC1 = ACC1 + QIN( 1, I )*DELTA( I )
                  ACC2 = ACC2 + QIN( 2, I )*DELTA( I )
   10          CONTINUE
               QOUT( 1, J ) = ACC1
               QOUT( 2, J ) = ACC2
            ELSE
               DO 20 R = 1, NBR
                  ROWSUM = ZERO
                  DO 15 I = 1, K
                     ROWSUM = ROWSUM + QIN( R, I )*DELTA( I )
   15             CONTINUE
                  QOUT( R, J ) = ROWSUM
   20          CONTINUE
            END IF
   30    CONTINUE
         GO TO 170
      END IF
*
*     Preserve the update vector, solve all secular roots, then accumulate
*     the product needed to reconstruct DLAED9's updated W.  In the
*     parallel path each thread owns both a DELTA buffer and a WPROD
*     accumulator, so the high-relative-accuracy DELTA values produced by
*     DLAED4_BR are consumed immediately rather than reconstructed later
*     from the compact origin/tau representation.
*
      DO 40 I = 1, K
         WSGN( I ) = W( I )
   40 CONTINUE
*
      NTHREADS = 1
!$    NTHREADS = BRDC_WPROD_NTHREADS()
      IF( NTHREADS.LE.1 ) THEN
         DO 41 I = 1, K
            WPROD( I ) = ONE
   41    CONTINUE
*
         DO 44 J = 1, K
            CALL DLAED4_BR( K, J, DLAMBDA, WSGN, DELTA, RHO, D( J ),
     $                      IORG, TAU( J ), NEAR( 1, J ),
     $                      NEAR( 2, J ), INFO )
            IF( INFO.NE.0 )
     $         GO TO 170
*
            WPROD( J ) = WPROD( J )*DELTA( J )
            DO 42 I = 1, J - 1
               WPROD( I ) = WPROD( I )*
     $                      ( DELTA( I ) /
     $                        ( DLAMBDA( I )-DLAMBDA( J ) ) )
   42       CONTINUE
            DO 43 I = J + 1, K
               WPROD( I ) = WPROD( I )*
     $                      ( DELTA( I ) /
     $                        ( DLAMBDA( I )-DLAMBDA( J ) ) )
   43       CONTINUE
   44    CONTINUE
      ELSE
         DO 45 TID = 0, NTHREADS - 1
            IOFF = TID*K
            DO 46 I = 1, K
               WACC( IOFF+I ) = ONE
   46       CONTINUE
   45    CONTINUE
*
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(I,J,TID,IOFF,IORG,
!$OMP& INFOJ)
!$OMP& IF(K.GT.128) NUM_THREADS(NTHREADS) SCHEDULE(STATIC)
         DO 70 J = 1, K
            TID = 0
            INFOJ = 0
!$          TID = OMP_GET_THREAD_NUM()
            IOFF = TID*K
            CALL DLAED4_BR( K, J, DLAMBDA, WSGN, DWORK( IOFF+1 ),
     $                      RHO, D( J ), IORG, TAU( J ),
     $                      NEAR( 1, J ), NEAR( 2, J ), INFOJ )
            IF( INFOJ.EQ.0 ) THEN
               WACC( IOFF+J ) = WACC( IOFF+J )*
     $                          DWORK( IOFF+J )
               DO 47 I = 1, J - 1
                  WACC( IOFF+I ) = WACC( IOFF+I )*
     $                             ( DWORK( IOFF+I ) /
     $                               ( DLAMBDA( I )-DLAMBDA( J ) ) )
   47          CONTINUE
               DO 48 I = J + 1, K
                  WACC( IOFF+I ) = WACC( IOFF+I )*
     $                             ( DWORK( IOFF+I ) /
     $                               ( DLAMBDA( I )-DLAMBDA( J ) ) )
   48          CONTINUE
            END IF
            IF( INFOJ.NE.0 ) THEN
!$OMP CRITICAL
               IF( INFO.EQ.0 )
     $            INFO = INFOJ
!$OMP END CRITICAL
            END IF
   70    CONTINUE
!$OMP END PARALLEL DO
         IF( INFO.NE.0 )
     $      GO TO 170
*
         DO 75 I = 1, K
            TEMP = ONE
            DO 76 TID = 0, NTHREADS - 1
               TEMP = TEMP*WACC( TID*K+I )
   76       CONTINUE
            WPROD( I ) = TEMP
   75    CONTINUE
      END IF
*
      DO 80 I = 1, K
         W( I ) = SIGN( SQRT( -WPROD( I ) ), WSGN( I ) )
   80 CONTINUE
*
*     Stream each secular eigenvector column through the selected rows.
*     Reconstruct the DELTA column from the compact origin/tau state,
*     using cached pole-adjacent values where cancellation is largest.
*
      IF( NBR.EQ.2 ) THEN
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(J,I,IORG,N1,N2,TEMP,
!$OMP& ACC1,ACC2,DLT)
!$OMP& IF(K.GT.128) NUM_THREADS(NTHREADS) SCHEDULE(STATIC)
         DO 115 J = 1, K
            IF( J.EQ.K ) THEN
               IORG = K
               N1 = K - 1
               N2 = K
            ELSE IF( TAU( J ).GE.ZERO ) THEN
               IORG = J
               N1 = J
               N2 = J + 1
            ELSE
               IORG = J + 1
               N1 = J
               N2 = J + 1
            END IF
            TEMP = ZERO
            ACC1 = ZERO
            ACC2 = ZERO
*
            DO 85 I = 1, N1 - 1
               DLT = W( I ) /
     $               ( ( DLAMBDA( I )-DLAMBDA( IORG ) ) - TAU( J ) )
               TEMP = TEMP + DLT*DLT
               ACC1 = ACC1 + QIN( 1, I )*DLT
               ACC2 = ACC2 + QIN( 2, I )*DLT
   85       CONTINUE
*
            DLT = W( N1 ) / NEAR( 1, J )
            TEMP = TEMP + DLT*DLT
            ACC1 = ACC1 + QIN( 1, N1 )*DLT
            ACC2 = ACC2 + QIN( 2, N1 )*DLT
*
            DO 86 I = N1 + 1, N2 - 1
               DLT = W( I ) /
     $               ( ( DLAMBDA( I )-DLAMBDA( IORG ) ) - TAU( J ) )
               TEMP = TEMP + DLT*DLT
               ACC1 = ACC1 + QIN( 1, I )*DLT
               ACC2 = ACC2 + QIN( 2, I )*DLT
   86       CONTINUE
*
            DLT = W( N2 ) / NEAR( 2, J )
            TEMP = TEMP + DLT*DLT
            ACC1 = ACC1 + QIN( 1, N2 )*DLT
            ACC2 = ACC2 + QIN( 2, N2 )*DLT
*
            DO 87 I = N2 + 1, K
               DLT = W( I ) /
     $               ( ( DLAMBDA( I )-DLAMBDA( IORG ) ) - TAU( J ) )
               TEMP = TEMP + DLT*DLT
               ACC1 = ACC1 + QIN( 1, I )*DLT
               ACC2 = ACC2 + QIN( 2, I )*DLT
   87       CONTINUE
*
            TEMP = SQRT( TEMP )
            QOUT( 1, J ) = ACC1 / TEMP
            QOUT( 2, J ) = ACC2 / TEMP
  115    CONTINUE
!$OMP END PARALLEL DO
         GO TO 170
      END IF
*
      DO 130 J = 1, K
         IF( J.EQ.K ) THEN
            IORG = K
         ELSE IF( TAU( J ).GE.ZERO ) THEN
            IORG = J
         ELSE
            IORG = J + 1
         END IF
         TEMP = ZERO
         IF( NBR.EQ.2 ) THEN
            ACC1 = ZERO
            ACC2 = ZERO
         ELSE
            DO 90 R = 1, NBR
               QOUT( R, J ) = ZERO
   90       CONTINUE
         END IF
         DO 100 I = 1, K
            DLT = ( DLAMBDA( I )-DLAMBDA( IORG ) ) - TAU( J )
            IF( J.EQ.K ) THEN
               IF( I.EQ.K-1 )
     $            DLT = NEAR( 1, J )
               IF( I.EQ.K )
     $            DLT = NEAR( 2, J )
            ELSE
               IF( I.EQ.J )
     $            DLT = NEAR( 1, J )
               IF( I.EQ.J+1 )
     $            DLT = NEAR( 2, J )
            END IF
            DLT = W( I ) / DLT
            TEMP = TEMP + DLT*DLT
            IF( NBR.EQ.2 ) THEN
               ACC1 = ACC1 + QIN( 1, I )*DLT
               ACC2 = ACC2 + QIN( 2, I )*DLT
            ELSE
               DO 95 R = 1, NBR
                  QOUT( R, J ) = QOUT( R, J ) + QIN( R, I )*DLT
   95          CONTINUE
            END IF
  100    CONTINUE
         TEMP = SQRT( TEMP )
         IF( NBR.EQ.2 ) THEN
            QOUT( 1, J ) = ACC1 / TEMP
            QOUT( 2, J ) = ACC2 / TEMP
         ELSE
            DO 120 R = 1, NBR
               QOUT( R, J ) = QOUT( R, J ) / TEMP
  120       CONTINUE
         END IF
  130 CONTINUE
*
  170 CONTINUE
      RETURN
*
*     End of DLAED9_BR
*
      END

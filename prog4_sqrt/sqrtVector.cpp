#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <thread>
#include <immintrin.h>


typedef struct {
    float *values;
    float initialGuess;
    float *output;
    float size;
} WorkerArgs;

void threadStart(WorkerArgs * const args) {
    __m256 curVals;
    __m256 guessVec;
    __m256 onesVec;
    __m256 negOnesVec;
    __m256 errorVec;
    __m256 negativeErrorVec;
    __m256 zerosVec;
    __m256 thresholdVec; 
    __m256 tempGuess;
    __m256 threesVec;
    __m256 halfsVec;
    __m256 newOutputsVec;
    __m256 oldOutputsVec;

    __m256 negativeErrorMask;
    __m256 thresholdMask;
    __m256 belowThresholdMask;

    for (int i=0; i<args->size; i+=8) {
        curVals = _mm256_load_ps(args->values + i);
        guessVec = _mm256_set1_ps(args->initialGuess);
        onesVec = _mm256_set1_ps(1.f);
        zerosVec = _mm256_set1_ps(0.f);
        negOnesVec = _mm256_set1_ps(-1.f);
        threesVec = _mm256_set1_ps(3.f);
        halfsVec = _mm256_set1_ps(0.5f);

        errorVec = _mm256_mul_ps(guessVec, guessVec);
        errorVec = _mm256_mul_ps(curVals, errorVec);
        errorVec = _mm256_sub_ps(errorVec, onesVec);

        negativeErrorMask = _mm256_cmp_ps(errorVec, zerosVec, _CMP_LT_OQ);
        negativeErrorVec = _mm256_mul_ps(errorVec, negOnesVec);
        errorVec = _mm256_blendv_ps(errorVec, negativeErrorVec, negativeErrorMask);
        thresholdVec = _mm256_set1_ps(0.00001f);

        thresholdMask = _mm256_cmp_ps(errorVec, thresholdVec, _CMP_GT_OQ);
        belowThresholdMask = _mm256_cmp_ps(errorVec, thresholdVec, _CMP_LE_OQ);
        oldOutputsVec = _mm256_set1_ps(0.f);
        newOutputsVec = _mm256_mul_ps(curVals, guessVec);
        oldOutputsVec = _mm256_blendv_ps(oldOutputsVec, newOutputsVec, belowThresholdMask);
       
        int maskCompressed = _mm256_movemask_ps(thresholdMask);
        int numSet = _mm_popcnt_u32(maskCompressed);

        while (numSet > 0) {
            tempGuess = _mm256_mul_ps(_mm256_mul_ps(guessVec, guessVec), guessVec);
            tempGuess = _mm256_mul_ps(tempGuess, curVals);
            guessVec = _mm256_mul_ps(threesVec, guessVec);

            guessVec = _mm256_sub_ps(guessVec, tempGuess);
            guessVec = _mm256_mul_ps(guessVec, halfsVec);

            errorVec = _mm256_mul_ps(guessVec, guessVec);
            errorVec = _mm256_mul_ps(curVals, errorVec);
            errorVec = _mm256_sub_ps(errorVec, onesVec);

            negativeErrorMask = _mm256_cmp_ps(errorVec, zerosVec, _CMP_LT_OQ);
            negativeErrorVec = _mm256_mul_ps(errorVec, negOnesVec);
            errorVec = _mm256_blendv_ps(errorVec, negativeErrorVec, negativeErrorMask);
            thresholdVec = _mm256_set1_ps(0.00001f);

            thresholdMask = _mm256_cmp_ps(errorVec, thresholdVec, _CMP_GT_OQ);
            belowThresholdMask = _mm256_cmp_ps(errorVec, thresholdVec, _CMP_LE_OQ);
            newOutputsVec = _mm256_mul_ps(curVals, guessVec);
            oldOutputsVec = _mm256_blendv_ps(oldOutputsVec, newOutputsVec, belowThresholdMask);

            maskCompressed = _mm256_movemask_ps(thresholdMask);
            numSet = _mm_popcnt_u32(maskCompressed);
        } 

        _mm256_store_ps(args->output + i, oldOutputsVec);

    }

}

void sqrtVector(int N,
                float initialGuess,
                float values[],
                float output[])
{
    // make 64 threads 
    // on each thread, do SIMD
    std::thread workers[64];
    WorkerArgs args[64];

    int threadNum = 0;
    for (int i = 0; i < N; i+= N / 64) {
        std::copy(values, values + N / 64, args[threadNum].values);
        args[threadNum].initialGuess = initialGuess;
        args[threadNum].output = output;
        args[threadNum].size = N / 64;
        threadNum++;
    }

    for (int i = 1; i < 64; i++) {
        workers[i] = std::thread(threadStart, &args[i]);
    }

    threadStart(&args[0]);

    for (int i = 1; i < 64; i++) {
        workers[i].join();
    }

}


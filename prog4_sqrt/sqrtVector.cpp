#include <algorithm>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <thread>
#include <immintrin.h>


typedef struct {
    int start, end;
    float *values;
    float initialGuess;
    float *output;
} WorkerArgs;

__m256 ones = _mm256_set1_ps(1.f);

__m256 getAbsError(__m256 values, __m256 guess) {
    __m256 errors = 
        _mm256_mul_ps(_mm256_mul_ps(guess, guess), values);
    errors = _mm256_sub_ps(errors, ones);

    return _mm256_andnot_ps(_mm256_set1_ps(-0.0f), errors);
}

int getNumSet(__m256 mask) {
    int maskCompressed = _mm256_movemask_ps(mask);
    return _mm_popcnt_u32(maskCompressed);
}

void threadStart(WorkerArgs * const args) {

    // constants
    __m256 threes = _mm256_set1_ps(3.f);
    __m256 halfs = _mm256_set1_ps(0.5f);
    __m256 thresholds = _mm256_set1_ps(0.00001f);

    // value vectors
    __m256 curVals;
    __m256 guessVec;
    __m256 errorVec;
    __m256 tempGuess;
    __m256 outputs;

    // masks
    __m256 thresholdMask;
    __m256 belowMask;

    for (int i = args->start; i < args->end; i += 8) {
        // load new values and initial guess
        curVals = _mm256_loadu_ps(args->values + i);
        guessVec = _mm256_set1_ps(args->initialGuess);

        // compute error from guess
        errorVec = getAbsError(curVals, guessVec);
        outputs = _mm256_mul_ps(curVals, guessVec);

        thresholdMask = _mm256_cmp_ps(errorVec, thresholds, _CMP_GT_OQ);
        belowMask = _mm256_cmp_ps(errorVec, thresholds, _CMP_LE_OQ);
        int numSet = getNumSet(thresholdMask);
        
        while (numSet > 0) {
            tempGuess = _mm256_mul_ps(_mm256_mul_ps(guessVec, guessVec), guessVec);
            tempGuess = _mm256_mul_ps(tempGuess, curVals);
            guessVec = _mm256_mul_ps(threes, guessVec);
            guessVec = _mm256_sub_ps(guessVec, tempGuess);
            guessVec = _mm256_mul_ps(guessVec, halfs);

            errorVec = getAbsError(curVals, guessVec);
            outputs = _mm256_blendv_ps(_mm256_mul_ps(curVals, guessVec),
                                       outputs,
                                       belowMask);

            belowMask = _mm256_or_ps(_mm256_cmp_ps(errorVec, thresholds, _CMP_LE_OQ),
                                     belowMask);
            thresholdMask = _mm256_cmp_ps(errorVec, thresholds, _CMP_GT_OQ);
            numSet = getNumSet(thresholdMask);
        } 

        if (i + 8 >= args->end) {
            float temp[8];
            _mm256_storeu_ps(temp, outputs);
            for (int k = i; k < args->end; k++) {
                args->output[k] = temp[k-i];
            }
        } else {
            _mm256_storeu_ps(&args->output[i], outputs);
        }

    } 

}

void sqrtVector(int N,
                float initialGuess,
                float values[],
                float output[])
{
    int numThreads = 64;
    std::thread workers[numThreads];
    WorkerArgs args[numThreads];

    for (int i = 0; i < numThreads; i++) {
        args[i].values = values;
        args[i].initialGuess = initialGuess;
        args[i].output = output;
        args[i].start = i * N / numThreads;
        args[i].end =
            args[i].start + N / numThreads;

        if (i == numThreads - 1) {
            args[i].end += N % numThreads;
        }
    } 

    for (int i = 1; i < numThreads; i++) {
        workers[i] = std::thread(threadStart, &args[i]);
    }

    threadStart(&args[0]);

    for (int i = 1; i < numThreads; i++) {
        workers[i].join();
    }
}


# =============================================================================
# config.py — Centralized constants for all exercises
# =============================================================================

# --- Pixel to CM conversion ---
# Approximate ratio at 1 meter distance
SIT_AND_REACH_PIXEL_TO_CM = 0.533333
BACK_SCRATCH_PIXEL_TO_CM  = 0.625

# --- Sit and Reach: angle boundaries ---
# Calibration phase
SAR_CALIB_ELBOW_MIN      = 20
SAR_CALIB_ELBOW_MAX      = 120
SAR_CALIB_HIP_MIN        = 120
SAR_CALIB_HIP_MAX        = 160
SAR_CALIB_KNEE_MIN       = 100
SAR_CALIB_KNEE_MAX       = 180

# Posture / measurement phase
SAR_POSTURE_ELBOW_MIN    = 155
SAR_POSTURE_ELBOW_MAX    = 180
SAR_POSTURE_HIP_MIN      = 55
SAR_POSTURE_HIP_MAX      = 150
SAR_POSTURE_KNEE_MIN     = 100
SAR_POSTURE_KNEE_MAX     = 180

# Opposite-side limits (shared between phases)
SAR_OPP_ELBOW_MIN        = 140
SAR_OPP_ELBOW_MAX        = 180
SAR_OPP_KNEE_MIN         = 80
SAR_OPP_KNEE_MAX         = 150

# --- Sit and Reach: timing & averaging ---
SAR_CALIBRATION_DURATION = 3    # seconds to hold calibration pose
SAR_POSE_DURATION        = 3    # seconds to hold measurement pose
SAR_AVERAGE_OVER         = 6    # rolling window for distance smoothing
SAR_ERROR                = 1.035  # systematic error correction (cm)

# --- Back Scratch: thresholds ---
BS_DISTANCE_THRESHOLD    = 33   # cm — hands must be closer than this
BS_POSE_HELD_DURATION    = 3    # seconds to hold final pose
BS_POSE_NO_HELD_DURATION = 1.5  # seconds before timer resets on lost detection
BS_AVERAGE_OVER          = 5
BS_ERROR                 = 1.91  # systematic error correction (cm)

# --- MediaPipe pose landmark indices (for reference) ---
# 11=L_shoulder  12=R_shoulder  13=L_elbow   14=R_elbow
# 15=L_wrist     16=R_wrist     19=L_index   20=R_index
# 23=L_hip       24=R_hip       25=L_knee    26=R_knee
# 27=L_ankle     28=R_ankle     29=L_heel    30=R_heel
# 31=L_foot_idx  32=R_foot_idx
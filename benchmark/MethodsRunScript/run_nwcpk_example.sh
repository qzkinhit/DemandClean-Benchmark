#!/bin/bash
# Make executable: chmod +x CleanerRunScript/run_raha_baran/run_nwcpk.sh
# Run with: ./CleanerRunScript/run_raha_baran/run_nwcpk.sh

# Dataset configuration
# Each entry encodes: dataset index_attr mse_attr noise_dir clean_path. Adjust paths to your setup.
datasets=(
    "1_hospitals:index:Score:Data/1_hospitals/noise_with_correct_primary_key:Data/1_hospitals/clean_index.csv"
    "2_flights:index::Data/2_flights/noise_with_correct_primary_key:Data/2_flights/clean_index.csv"
    "3_beers:id:abv ibu:Data/3_beers/noise_with_correct_primary_key:Data/3_beers/clean.csv"
    "4_rayyan:index::Data/4_rayyan/noise_with_correct_primary_key:Data/4_rayyan/clean_index.csv"
    "5_tax_20k:tno:rate:Data/5_tax/tax_20k/noise_with_correct_primary_key:Data/5_tax/tax_20k/tax_20k_clean_id.csv"
    "5_tax_50k:tno:rate:Data/5_tax/tax_50k/noise_with_correct_primary_key:Data/5_tax/tax_50k/tax_50k_clean_id.csv"
    "5_tax_200k:tno:rate:Data/5_tax/tax_200k/noise_with_correct_primary_key:Data/5_tax/tax_200k/tax_200k_clean_id.csv"
    "6_soccer:index::Data/6_soccer/noise_with_correct_primary_key:Data/6_soccer/clean_index.csv"
)

# Error-ratio set
error_ratios=("0.25" "0.5" "0.75" "1" "1.25" "1.5" "1.75" "2")
#error_ratios=("0.25")

# Create log directory
log_dir="logs/raha_baran_nwcpk"
mkdir -p "${log_dir}"

# Iterate over datasets and error ratios; generate and run the command (tune for your system)
for dataset_config in "${datasets[@]}"; do
    # Parse key/value pairs
    IFS=":" read -r dataset index_attr mse_attr noise_dir clean_path <<< "${dataset_config}"
    # Dataset short name starts at the 3rd character
    short_dataset_name="${dataset:2}"
    for ratio in "${error_ratios[@]}"; do
        task_name="${dataset}_nwcpk_${ratio//./}"
        dirty_path="${noise_dir}/dirty_mixed_${ratio}/dirty_${short_dataset_name}_mix_${ratio}.csv"
        output_path="results/raha_baran/nwcpk"
        log_file="${log_dir}/${dataset}_raha_baran_nwcpk_${ratio//./}.log"

        # Build command
        cmd="python3 CleanerRunScript/run_raha_baran/repair_with_raha.py --dirty_path ${dirty_path} --clean_path ${clean_path} --task_name ${task_name} --output_path ${output_path} --index_attribute ${index_attr}"

        if [ -n "${mse_attr}" ]; then
            cmd+=" --mse_attributes ${mse_attr}"
        fi

        # Echo the command for debugging
        echo "Generated command:"
        echo "${cmd}"

        # Run it
        eval "${cmd}" &> "${log_file}"

        if [ $? -ne 0 ]; then
            echo "Error: Command failed for task ${task_name}. Check ${log_file} for details."
            exit 1
        fi
        echo "Task ${task_name} completed successfully. Log saved to: ${log_file}"
    done
done

echo "All Raha Baran tasks completed successfully."
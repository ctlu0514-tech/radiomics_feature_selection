import os
import pandas as pd
import SimpleITK as sitk
import radiomics
from radiomics import featureextractor
import SimpleITK as sitk

# --- 1. 配置您的路径 ---
# !! 警告: 请根据您的系统修改这两个路径 !!
# BASE_PATH 是指向 'dataset' 文件夹的 *父* 文件夹
# (根据您的截图，'dataset' 位于 /nfs/zc1/qianliexian/ 下)
BASE_PATH = '/nfs/zc1/qianliexian'
# OUTPUT_CSV 是您希望保存最终特征表格的地方
OUTPUT_CSV = '/data/qh_20T_share_file/lct/CT67/qianliexian_all_radiomics_features.csv'
# -------------------------


def get_base_settings():
    """
    所有模态共享的基础设置 (重采样)
    """
    settings = {}
    # 关键：将所有图像重采样到相同的体素间距 (1x1x1 mm)
    settings['Interpolator'] = sitk.sitkBSpline
    settings['resampledPixelSpacing'] = [1, 1, 1]
    
    # --- 新增修复 ---
    # 关键: 如果图像和掩码的空间信息(原点,间距,方向)不匹配，
    # 强制将掩码重采样到(重采样后的)图像网格上，而不是报错。
    # 这将修复 "Bounding box of ROI is larger than image space" 错误。
    settings['correctMask'] = True
    # ---------------
    
    # 启用所有特征和所有图像类型 (LoG, Wavelet等)
    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    extractor.enableAllImageTypes()
    extractor.enableAllFeatures()
    
    # 手动启用默认禁用的一阶特征和形状特征
    extractor.enableFeaturesByName(firstorder=[]) # 启用所有一阶
    extractor.enableFeaturesByName(shape=[])      # 启用所有形状
    
    return extractor.settings


def get_mri_extractor():
    """
    获取 MRI (T2, DWI, ADC) 的专用提取器
    - 必须归一化
    - 使用 binCount (分箱计数) 而不是 binWidth
    """
    print("初始化 MRI (T2/DWI/ADC) 提取器...")
    settings = get_base_settings()
    
    # 关键: MRI 必须归一化
    settings['normalize'] = True
    settings['normalizeScale'] = 100
    
    # 关键: MRI 使用 binCount，而不是 binWidth
    settings['binCount'] = 32 
    
    # 从设置中移除 binWidth (如果存在)，以避免冲突
    settings.pop('binWidth', None) 
    
    return featureextractor.RadiomicsFeatureExtractor(**settings)


def get_ct_extractor():
    """
    获取 CT 专用提取器
    - 归一化 (用于处理负的HU值)
    - 使用固定的 binWidth
    """
    print("初始化 CT 提取器...")
    settings = get_base_settings()
    
    settings['normalize'] = True
    settings['normalizeScale'] = 100
    # 关键: CT 使用 voxelArrayShift 来处理负的 HU 值
    settings['voxelArrayShift'] = 1000 
    
    # 关键: CT 使用固定的 binWidth
    settings['binWidth'] = 25 
    
    return featureextractor.RadiomicsFeatureExtractor(**settings)


def get_pet_extractor():
    """
    获取 PET 专用提取器
    - 不归一化 (SUV 是绝对值)
    - 使用固定的、小的 binWidth
    """
    print("初始化 PET 提取器...")
    settings = get_base_settings()
    
    # 关键: PET (SUV) 不应归一化
    settings['normalize'] = False 
    
    # 关键: PET 使用一个固定的、适合 SUV 范围的小 binWidth
    settings['binWidth'] = 0.25 # 常见的 PET binWidth
    
    return featureextractor.RadiomicsFeatureExtractor(**settings)


def extract_features(extractor, image_path, mask_path, modality_prefix):
    """
    一个通用的特征提取函数，包含错误处理、特征重命名和强制对齐
    """
    # 1. 检查文件是否存在
    if image_path is None or mask_path is None or not os.path.exists(image_path) or not os.path.exists(mask_path):
        print(f"    [警告] 跳过 {modality_prefix}. 缺少文件:")
        if image_path is None or not os.path.exists(image_path):
            print(f"        - 图像: {image_path} (未找到或路径为None)")
        if mask_path is None or not os.path.exists(mask_path):
            print(f"        - 掩码: {mask_path} (未找到或路径为None)")
        return None

    try:
        # --- 新增的关键修复步骤：强制重采样 ---
        # 1. 加载图像和掩码
        image = sitk.ReadImage(image_path)
        mask = sitk.ReadImage(mask_path)
        
        # 2. 创建一个重采样器
        resampler = sitk.ResampleImageFilter()
        
        # 3. 设置输出参数为目标图像的参数
        #    这保证了输出的掩码与图像有完全相同的原点、间距、尺寸和方向
        resampler.SetReferenceImage(image)
        
        # 4. 设置插值方法为"最近邻"，这对于标签掩码至关重要
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        
        # 5. 执行重采样
        corrected_mask = resampler.Execute(mask)
        # -----------------------------------------

        # 2. 提取特征
        #    !! 关键改动: 不再传递文件路径，而是传递SimpleITK的对象 !!
        #    传递修正后的掩码对象 (corrected_mask)，而不是原始路径
        result = extractor.execute(image, corrected_mask, label=1)
        
        prefixed_result = {}
        for key, val in result.items():
            if not key.startswith('diagnostics'):
                prefixed_result[f"{modality_prefix}_{key}"] = val
        
        print(f"    - {modality_prefix}: 提取成功")
        return prefixed_result
        
    except Exception as e:
        print(f"    [!!错误!!] 提取 {modality_prefix} 失败. 图像: {image_path}, 掩码: {mask_path}")
        print(f"     错误信息: {e}")
        return None


def find_file_by_parts(directory, prefix, suffix):
    """
    在目录中查找以特定前缀和后缀结尾的第一个文件。
    """
    if not os.path.exists(directory):
        return None
    
    for filename in os.listdir(directory):
        if filename.startswith(prefix) and filename.endswith(suffix):
            return os.path.join(directory, filename)
    
    print(f"    [调试] 在 {directory} 中未找到匹配 {prefix}*{suffix} 的文件")
    return None # 未找到


def process_all_patients():
    """
    主函数：遍历所有患者并提取所有模态的特征
    """
    dataset_path = os.path.join(BASE_PATH, 'dataset')
    
    # 初始化所有提取器
    mri_extractor = get_mri_extractor()
    ct_extractor = get_ct_extractor()
    pet_extractor = get_pet_extractor()
    
    all_features_list = [] # 存储所有患者的特征

    # 我们通过扫描一个目录来获取所有患者ID (例如 'mpMri_nii')
    patient_id_dir = os.path.join(dataset_path, 'mpMri_nii')
    if not os.path.exists(patient_id_dir):
        print(f"错误: 找不到患者目录 {patient_id_dir}")
        print("请检查您的 BASE_PATH 是否设置正确。")
        return

    patient_ids = sorted([d for d in os.listdir(patient_id_dir) 
                          if os.path.isdir(os.path.join(patient_id_dir, d))])
    
    print(f"\n找到了 {len(patient_ids)} 个患者. 开始处理...")

    for patient_id in patient_ids:
        print(f"--- 正在处理患者: {patient_id} ---")
        
        # patient_features 字典将存储 *这一个患者* 的 *所有* 模态特征
        patient_features = {'patient_id': patient_id}
        
        # --- 1. 处理 mpMRI (T2, DWI, ADC) ---
        mri_image_dir = os.path.join(dataset_path, 'mpMri_nii', patient_id)
        mri_mask_dir = os.path.join(dataset_path, 'mpMRI', patient_id)
        mri_modalities = ['T2', 'DWI', 'ADC']
        
        for modality in mri_modalities:
            # 构建文件路径 (使用新的查找逻辑)
            image_path = find_file_by_parts(mri_image_dir, patient_id, 
                                            f"{modality}.nii.gz")
            mask_path = find_file_by_parts(mri_mask_dir, patient_id, 
                                           f"{modality}_Merge.nii")
            
            # 提取特征
            features = extract_features(mri_extractor, image_path, mask_path, 
                                        modality_prefix=modality)
            if features:
                patient_features.update(features)

        # --- 2. 处理 PET/CT (CT, PET) ---
        petct_image_dir = os.path.join(dataset_path, 'PETCT_nii', patient_id)
        petct_mask_dir = os.path.join(dataset_path, 'PETCT', patient_id)

        # 2a. 查找 CT 图像
        ct_image_path = find_file_by_parts(petct_image_dir, patient_id, 
                                           "_CT.nii.gz")
        # 2b. 查找 CT 对应的掩码 (例如 1002XIANGGRUIPING.nii.gz)
        #    这是我们将用于 CT 和 PET 的 *唯一* 掩码
        ct_mask_path = find_file_by_parts(petct_mask_dir, patient_id,
                                          ".nii.gz")

        # 2c. 查找 PET 图像
        pet_image_path = find_file_by_parts(petct_image_dir, patient_id, 
                                            "_PET.nii.gz")
        
        # 2d. 查找 PET 对应的掩码 (例如 1002pet.nii)
        #    !! 我们不再使用这个文件了，因为它空间不一致 !!
        pet_mask_path = find_file_by_parts(petct_mask_dir, patient_id, 
                                             "pet.nii")

        # 2e. 提取 CT 特征 (使用 CT 图像 + CT 掩码)
        ct_features = extract_features(ct_extractor, ct_image_path, ct_mask_path, 
                                       modality_prefix='CT')
        if ct_features:
            patient_features.update(ct_features)
            
        # 2f. 提取 PET 特征 (使用 PET 图像 + CT 掩码)
        #    !! 关键改动: 这里使用 ct_mask_path，而不是 pet_mask_path !!
        pet_features = extract_features(pet_extractor, pet_image_path, pet_mask_path, 
                                        modality_prefix='PET')
        if pet_features:
            patient_features.update(pet_features)
            
        # --- 3. 收集结果 ---
        # 只有在提取到至少一个特征时才添加 (避免只有id的空行)
        if len(patient_features) > 1:
            all_features_list.append(patient_features)

    # --- 4. 保存到 CSV ---
    if not all_features_list:
        print("未提取到任何特征。请检查您的路径和文件命名。")
        return

    print("\n--- 所有患者处理完毕. 正在保存到 CSV... ---")
    try:
        df = pd.DataFrame(all_features_list)
        # 将 patient_id 列移到最前面
        df = df[ ['patient_id'] + [ col for col in df.columns if col != 'patient_id' ] ]
        
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"\n🎉 成功! 特征已保存到: {OUTPUT_CSV}")
        print(f"总共处理了 {len(df)} 名患者。")
        print(f"总共提取了 {len(df.columns) - 1} 个特征 (已包含所有模态)。")
        
    except Exception as e:
        print(f"!! 错误: 无法保存 CSV 文件. {e}")
        print("您可能没有目标文件夹的写入权限。")


if __name__ == '__main__':
    # 设置 PyRadiomics 日志级别，减少不必要的输出
    radiomics.setVerbosity(40) # 40 = WARNING
    process_all_patients()
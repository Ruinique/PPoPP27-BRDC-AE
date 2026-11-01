#ifndef EVD_LIBRARY_INTERNAL_SBR_HISTORY_LAYOUT_HPP
#define EVD_LIBRARY_INTERNAL_SBR_HISTORY_LAYOUT_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "evd/types.h"
#include "workspace_align.hpp"

namespace evd::internal {

inline constexpr std::int64_t kSbrHistorySuperblockCols = 1024;

inline std::size_t sbr_history_align_up(
    const std::size_t value,
    const std::size_t alignment = kDeviceWorkspaceAlignment) {
    const std::size_t remainder = value % alignment;
    return remainder == 0 ? value : value + alignment - remainder;
}

struct SbrHistoryBlock {
    std::int64_t first_history_col{0};
    std::int64_t block_cols{0};
    std::int64_t first_row{0};
    std::int64_t rows{0};
    std::size_t offset_elements{0};
    std::size_t elements{0};
};

struct SbrHistoryLayout {
    std::int64_t n{0};
    std::int64_t band_size{0};
    std::int64_t superblock_cols{0};
    std::int64_t block_count{0};
    std::size_t matrix_elements{0};
    std::size_t matrix_bytes{0};
    std::size_t total_bytes{0};
};

inline std::int64_t sbr_history_superblock_cols(
    const evd_syev2stage_config &config) {
    if (config.band_size <= 0) {
        return 0;
    }
    return std::max(config.band_size,
                    std::min<std::int64_t>(kSbrHistorySuperblockCols,
                                           config.bulge_block_size));
}

inline SbrHistoryLayout
make_sbr_history_layout(const evd_syev2stage_config &config) {
    SbrHistoryLayout layout{};
    if (config.compute_eigenvectors == 0 || config.n <= 0 ||
        config.band_size <= 0) {
        return layout;
    }

    const std::int64_t usable_cols =
        ((config.n - 1) / config.band_size) * config.band_size;
    if (usable_cols <= 0) {
        return layout;
    }

    const std::int64_t super_cols = sbr_history_superblock_cols(config);
    if (super_cols <= 0) {
        return layout;
    }

    layout.n = config.n;
    layout.band_size = config.band_size;
    layout.superblock_cols = super_cols;

    std::size_t elements = 0;
    std::int64_t block_count = 0;
    for (std::int64_t block_start = 0; block_start < usable_cols;
         block_start += super_cols) {
        const std::int64_t block_cols =
            std::min(super_cols, usable_cols - block_start);
        const std::int64_t first_row = block_start + config.band_size;
        const std::int64_t rows = config.n - first_row;
        if (block_cols <= 0 || rows <= 0) {
            continue;
        }
        elements += static_cast<std::size_t>(rows) *
                    static_cast<std::size_t>(block_cols);
        ++block_count;
    }

    layout.block_count = block_count;
    layout.matrix_elements = elements;
    layout.matrix_bytes = elements * sizeof(double);
    layout.total_bytes = sbr_history_align_up(2 * layout.matrix_bytes);
    return layout;
}

inline SbrHistoryBlock sbr_history_block_for_col(
    const SbrHistoryLayout &layout,
    const std::int64_t history_col) {
    SbrHistoryBlock block{};
    if (layout.n <= 0 || layout.band_size <= 0 ||
        layout.superblock_cols <= 0 || history_col < 0) {
        return block;
    }

    const std::int64_t usable_cols =
        ((layout.n - 1) / layout.band_size) * layout.band_size;
    if (history_col >= usable_cols) {
        return block;
    }

    std::size_t offset = 0;
    for (std::int64_t block_start = 0; block_start < usable_cols;
         block_start += layout.superblock_cols) {
        const std::int64_t block_cols =
            std::min(layout.superblock_cols, usable_cols - block_start);
        const std::int64_t first_row = block_start + layout.band_size;
        const std::int64_t rows = layout.n - first_row;
        const std::size_t elements =
            static_cast<std::size_t>(rows) *
            static_cast<std::size_t>(block_cols);
        if (history_col >= block_start &&
            history_col < block_start + block_cols) {
            block.first_history_col = block_start;
            block.block_cols = block_cols;
            block.first_row = first_row;
            block.rows = rows;
            block.offset_elements = offset;
            block.elements = elements;
            return block;
        }
        offset += elements;
    }
    return block;
}

inline SbrHistoryBlock sbr_history_block_for_col(
    const evd_syev2stage_config &config,
    const std::int64_t history_col) {
    return sbr_history_block_for_col(make_sbr_history_layout(config),
                                     history_col);
}

inline double *sbr_history_panel_ptr(
    double *base, const SbrHistoryBlock &block,
    const std::int64_t history_col) {
    if (base == nullptr || block.rows <= 0 || block.block_cols <= 0 ||
        history_col < block.first_history_col ||
        history_col >= block.first_history_col + block.block_cols) {
        return nullptr;
    }
    const std::int64_t local_col = history_col - block.first_history_col;
    const std::size_t offset = block.offset_elements +
                               static_cast<std::size_t>(local_col) *
                                   static_cast<std::size_t>(block.rows);
    return base + offset;
}

inline const double *sbr_history_panel_ptr(
    const double *base, const SbrHistoryBlock &block,
    const std::int64_t history_col) {
    return sbr_history_panel_ptr(const_cast<double *>(base), block,
                                 history_col);
}

} // namespace evd::internal

#endif // EVD_LIBRARY_INTERNAL_SBR_HISTORY_LAYOUT_HPP

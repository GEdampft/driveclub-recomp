// Reads single files out of the game's own archive (game.ndx + gameNNN.dat).
#pragma once
#include <cstdint>
#include <string>
#include <vector>

// Finds `path` (for example "newdata/vehicles.csv") in the archive whose index is in `update_dir` (or, when
// missing there, `base_dir`). The volume files are taken from `update_dir` first, then `base_dir`.
// Returns "" on success, otherwise a short description of what went wrong.
std::string DatxExtract(const std::wstring& update_dir, const std::wstring& base_dir, const std::string& path,
                        std::vector<uint8_t>& out);

// Which archive volumes (gameNNN.dat) the index uses, and which of them exist in neither folder. Returns "" on
// success (an empty `missing` means the dump is complete), otherwise what went wrong.
// `truncated` (when given) lists the volumes that exist but are shorter than their own chunk table says: an
// incomplete copy of the dump, which makes the game stall when it reads from the missing end.
std::string DatxCheckVolumes(const std::wstring& update_dir, const std::wstring& base_dir, unsigned& used,
                             std::vector<unsigned>& missing, std::vector<unsigned>* truncated = nullptr);

// Whether `path` is in the archive whose index is in `update_dir` (or, when missing there, `base_dir`).
bool DatxContains(const std::wstring& update_dir, const std::wstring& base_dir, const std::string& path);

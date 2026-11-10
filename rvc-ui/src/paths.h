// paths.h — runtime discovery of the repo root, the CLI tools and model folders.
//
// Resolution order:
//   root      : RVC_ROOT env var, else nearest ancestor of the executable
//               (or the current directory) containing a `models/` directory.
//   tools dir : RVC_TOOLS_DIR env var, else the executable's directory.
#pragma once

#include <QCoreApplication>
#include <QDir>
#include <QDirIterator>
#include <QFileInfo>
#include <QString>
#include <QStringList>

namespace paths {

#ifdef Q_OS_WIN
inline QString exe_suffix() { return QStringLiteral(".exe"); }
#else
inline QString exe_suffix() { return QString(); }
#endif

// Walk up from `start` (inclusive) until a directory containing `marker` is found.
inline QString find_up(const QString& start, const QString& marker, int max_levels = 6) {
    QDir dir(start);
    for (int i = 0; i < max_levels; ++i) {
        if (QFileInfo::exists(dir.filePath(marker))) {
            return dir.absolutePath();
        }
        if (!dir.cdUp()) {
            break;
        }
    }
    return {};
}

inline QString root() {
    const QString env = qEnvironmentVariable("RVC_ROOT");
    if (!env.isEmpty() && QFileInfo::exists(QDir(env).filePath(QStringLiteral("models")))) {
        return QDir(env).absolutePath();
    }
    const QString from_exe = find_up(QCoreApplication::applicationDirPath(), QStringLiteral("models"));
    if (!from_exe.isEmpty()) {
        return from_exe;
    }
    return find_up(QDir::currentPath(), QStringLiteral("models"));
}

inline QString tools_dir() {
    const QString env = qEnvironmentVariable("RVC_TOOLS_DIR");
    if (!env.isEmpty()) {
        return QDir(env).absolutePath();
    }
    return QCoreApplication::applicationDirPath();
}

inline QString tool_name(const QString& base) {
    return base + exe_suffix();
}

// Absolute path of a tool, or an empty string when it is not found.
inline QString tool(const QString& base) {
    const QString p = QDir(tools_dir()).filePath(tool_name(base));
    return QFileInfo::exists(p) ? p : QString();
}

// .gguf files directly inside `dir`, sorted by name.
inline QStringList gguf_files(const QString& dir) {
    QStringList out;
    for (const QFileInfo& fi : QDir(dir).entryInfoList({QStringLiteral("*.gguf")}, QDir::Files, QDir::Name)) {
        out << fi.absoluteFilePath();
    }
    return out;
}

inline QString models_root() { return QDir(root()).filePath(QStringLiteral("models")); }

// Search recursively for character models (all .gguf except *.index.gguf).
inline QStringList voice_models() {
    QStringList out;
    const QDir base(models_root() + QStringLiteral("/voice_models"));
    if (!base.exists()) {
        return out;
    }
    QDirIterator it(base.absolutePath(), {QStringLiteral("*.gguf")}, QDir::Files,
                    QDirIterator::Subdirectories);
    while (it.hasNext()) {
        const QString file = it.next();
        if (!file.endsWith(QStringLiteral(".index.gguf"))) {
            out << file;
        }
    }
    out.sort();
    return out;
}

// Picks the preferred entry from a list: the first containing `needle`, else the first.
inline QString preferred(const QStringList& list, const QString& needle) {
    for (const QString& item : list) {
        if (item.contains(needle, Qt::CaseInsensitive)) {
            return item;
        }
    }
    return list.isEmpty() ? QString() : list.first();
}

} // namespace paths

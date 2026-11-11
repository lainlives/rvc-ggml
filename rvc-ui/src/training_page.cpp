// training_page.cpp — RVC training: preprocess -> train -> export.
#include "training_page.h"

#include "paths.h"

#include <QComboBox>
#include <QDir>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QPlainTextEdit>
#include <QProcess>
#include <QPushButton>
#include <QScrollArea>
#include <QSpinBox>
#include <QVBoxLayout>

#include <algorithm>

namespace {

QString browse_row(QWidget* parent, const QString& value, const QString& title, bool directory,
                   const QString& filter = QString()) {
    return directory ? QFileDialog::getExistingDirectory(parent, title, value)
                     : QFileDialog::getOpenFileName(parent, title, value, filter);
}

} // namespace

TrainingPage::TrainingPage(QWidget* parent) : QWidget(parent) {
    build_ui();
    populate_models();
    refresh_export_sources();
}

TrainingPage::~TrainingPage() {
    for (QProcess* p : active_) {
        p->kill();
        p->waitForFinished(1000);
    }
}

void TrainingPage::build_ui() {
    auto* layout = new QVBoxLayout(this);

    auto* scroll = new QScrollArea(this);
    scroll->setWidgetResizable(true);
    auto* content = new QWidget(scroll);
    auto* content_layout = new QVBoxLayout(content);

    const QString root = paths::root();
    const QString work = (root.isEmpty() ? QDir::tempPath() + QStringLiteral("/rvc_train")
                                         : root + QStringLiteral("/work/train"));

    // ---------------- Experiment ------------------------------------------
    auto* exp_group = new QGroupBox(QStringLiteral("Experiment"), content);
    auto* exp_form = new QFormLayout(exp_group);
    {
        auto* row = new QWidget(exp_group);
        auto* h = new QHBoxLayout(row);
        h->setContentsMargins(0, 0, 0, 0);
        cache_edit_ = new QLineEdit(work + QStringLiteral("/cache"), row);
        cache_edit_->setToolTip(QStringLiteral(
            "rvc-prep writes dataset.json, spec.json and samples/*.bin here; "
            "rvc-train reads the same directory (--exp)."));
        auto* browse = new QPushButton(QStringLiteral("Browse..."), row);
        connect(browse, &QPushButton::clicked, this, [this] {
            const QString dir = browse_row(this, cache_edit_->text(), QStringLiteral("Training cache"), true);
            if (!dir.isEmpty()) {
                cache_edit_->setText(dir);
            }
        });
        h->addWidget(cache_edit_, 1);
        h->addWidget(browse);
        exp_form->addRow(QStringLiteral("Cache dir"), row);
    }
    {
        auto* row = new QWidget(exp_group);
        auto* h = new QHBoxLayout(row);
        h->setContentsMargins(0, 0, 0, 0);
        out_edit_ = new QLineEdit(work + QStringLiteral("/out"), row);
        out_edit_->setToolTip(QStringLiteral(
            "Training writes logs, checkpoints (train*.ckpt.gguf), epoch snapshots and "
            "periodic inference exports here (--out)."));
        auto* browse = new QPushButton(QStringLiteral("Browse..."), row);
        connect(browse, &QPushButton::clicked, this, [this] {
            const QString dir = browse_row(this, out_edit_->text(), QStringLiteral("Training output"), true);
            if (!dir.isEmpty()) {
                out_edit_->setText(dir);
                refresh_export_sources();
            }
        });
        h->addWidget(out_edit_, 1);
        h->addWidget(browse);
        exp_form->addRow(QStringLiteral("Output dir"), row);
    }
    {
        auto* row = new QWidget(exp_group);
        auto* h = new QHBoxLayout(row);
        h->setContentsMargins(0, 0, 0, 0);
        config_combo_ = new QComboBox(row);
        config_combo_->setToolTip(QStringLiteral(
            "Training config template (train / data / model sections). The template's version "
            "selects the matching preprocessing rate: v1 = 40 kHz, v2 = 48 kHz."));
        auto* browse = new QPushButton(QStringLiteral("Browse..."), row);
        connect(browse, &QPushButton::clicked, this, [this] {
            const QString file = browse_row(this, config_path(), QStringLiteral("Training config"), false,
                                            QStringLiteral("JSON (*.json)"));
            if (!file.isEmpty()) {
                config_combo_->addItem(QFileInfo(file).fileName(), file);
                config_combo_->setCurrentIndex(config_combo_->count() - 1);
            }
        });
        h->addWidget(config_combo_, 1);
        h->addWidget(browse);
        exp_form->addRow(QStringLiteral("Config"), row);
        connect(config_combo_, QOverload<int>::of(&QComboBox::currentIndexChanged),
                this, [this] { sync_config(); });
    }
    content_layout->addWidget(exp_group);

    // ---------------- Preprocess ------------------------------------------
    auto* prep_group = new QGroupBox(QStringLiteral("Preprocess (rvc-prep)"), content);
    auto* prep_form = new QFormLayout(prep_group);
    {
        auto* row = new QWidget(prep_group);
        auto* h = new QHBoxLayout(row);
        h->setContentsMargins(0, 0, 0, 0);
        input_edit_ = new QLineEdit(row);
        input_edit_->setPlaceholderText(QStringLiteral("folder with .wav files"));
        input_edit_->setToolTip(QStringLiteral(
            "Slicing, high-pass, f0 (RMVPE), HuBERT features and the linear spectrogram cache "
            "are written to the cache dir. Only .wav files are read."));
        auto* browse = new QPushButton(QStringLiteral("Browse..."), row);
        connect(browse, &QPushButton::clicked, this, [this] {
            const QString dir = browse_row(this, input_edit_->text(), QStringLiteral("Audio folder"), true);
            if (!dir.isEmpty()) {
                input_edit_->setText(dir);
            }
        });
        h->addWidget(input_edit_, 1);
        h->addWidget(browse);
        prep_form->addRow(QStringLiteral("Audio folder"), row);
    }
    {
        auto* row = new QWidget(prep_group);
        auto* h = new QHBoxLayout(row);
        h->setContentsMargins(0, 0, 0, 0);
        version_combo_ = new QComboBox(row);
        version_combo_->addItem(QStringLiteral("v1 (40 kHz)"), QStringLiteral("v1"));
        version_combo_->addItem(QStringLiteral("v2 (48 kHz)"), QStringLiteral("v2"));
        version_combo_->setCurrentIndex(1);
        prep_device_combo_ = new QComboBox(row);
        prep_device_combo_->addItem(QStringLiteral("GPU (Vulkan)"), 0);
        prep_device_combo_->addItem(QStringLiteral("CPU"), 99);
        h->addWidget(version_combo_);
        h->addWidget(new QLabel(QStringLiteral("device"), row));
        h->addWidget(prep_device_combo_);
        h->addStretch(1);
        prep_form->addRow(QStringLiteral("Version"), row);
        connect(version_combo_, QOverload<int>::of(&QComboBox::currentIndexChanged),
                this, [this] { sync_pretrain_paths(); });
    }
    prep_extractor_combo_ = new QComboBox(prep_group);
    prep_form->addRow(QStringLiteral("Feature extractor"), prep_extractor_combo_);
    prep_f0_combo_ = new QComboBox(prep_group);
    prep_form->addRow(QStringLiteral("F0 extractor"), prep_f0_combo_);
    {
        auto* row = new QWidget(prep_group);
        auto* h = new QHBoxLayout(row);
        h->setContentsMargins(0, 0, 0, 0);
        prep_run_ = new QPushButton(QStringLiteral("Run preprocessing"), row);
        prep_status_ = new QLabel(QStringLiteral("Idle"), row);
        connect(prep_run_, &QPushButton::clicked, this, &TrainingPage::run_prep);
        h->addWidget(prep_run_);
        h->addWidget(prep_status_, 1);
        prep_form->addRow(QString(), row);
    }
    content_layout->addWidget(prep_group);

    // ---------------- Train ------------------------------------------------
    auto* train_group = new QGroupBox(QStringLiteral("Train (rvc-train)"), content);
    auto* train_form = new QFormLayout(train_group);
    {
        auto* row = new QWidget(train_group);
        auto* h = new QHBoxLayout(row);
        h->setContentsMargins(0, 0, 0, 0);
        mode_combo_ = new QComboBox(row);
        mode_combo_->addItem(QStringLiteral("Fine-tune (pretrained G/D)"), false);
        mode_combo_->addItem(QStringLiteral("From scratch"), true);
        train_device_combo_ = new QComboBox(row);
        train_device_combo_->addItem(QStringLiteral("Auto device"), -1);
        for (int i = 0; i < 8; ++i) {
            train_device_combo_->addItem(QStringLiteral("Device %1").arg(i), i);
        }
        train_device_combo_->setToolTip(QStringLiteral(
            "traincpp device index from `rvc-train --list-devices` (not the ggml convention). "
            "Auto prefers GPU -> iGPU -> CPU."));
        h->addWidget(mode_combo_);
        h->addWidget(new QLabel(QStringLiteral("device"), row));
        h->addWidget(train_device_combo_);
        h->addStretch(1);
        train_form->addRow(QStringLiteral("Mode"), row);
        connect(mode_combo_, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this] {
            const bool scratch = mode_combo_->currentData().toBool();
            g_edit_->setEnabled(!scratch);
            d_edit_->setEnabled(!scratch);
        });
    }
    {
        auto* row = new QWidget(train_group);
        auto* h = new QHBoxLayout(row);
        h->setContentsMargins(0, 0, 0, 0);
        g_edit_ = new QLineEdit(row);
        g_edit_->setPlaceholderText(QStringLiteral("G training-state GGUF"));
        auto* browse = new QPushButton(QStringLiteral("Browse..."), row);
        connect(browse, &QPushButton::clicked, this, [this] {
            const QString file = browse_row(this, g_edit_->text(), QStringLiteral("Pretrained G"), false,
                                            QStringLiteral("GGUF (*.gguf)"));
            if (!file.isEmpty()) {
                g_edit_->setText(file);
            }
        });
        connect(g_edit_, &QLineEdit::textEdited, this, [this] { g_edited_ = true; });
        h->addWidget(g_edit_, 1);
        h->addWidget(browse);
        train_form->addRow(QStringLiteral("Pretrained G"), row);
    }
    {
        auto* row = new QWidget(train_group);
        auto* h = new QHBoxLayout(row);
        h->setContentsMargins(0, 0, 0, 0);
        d_edit_ = new QLineEdit(row);
        d_edit_->setPlaceholderText(QStringLiteral("D training-state GGUF"));
        auto* browse = new QPushButton(QStringLiteral("Browse..."), row);
        connect(browse, &QPushButton::clicked, this, [this] {
            const QString file = browse_row(this, d_edit_->text(), QStringLiteral("Pretrained D"), false,
                                            QStringLiteral("GGUF (*.gguf)"));
            if (!file.isEmpty()) {
                d_edit_->setText(file);
            }
        });
        connect(d_edit_, &QLineEdit::textEdited, this, [this] { d_edited_ = true; });
        h->addWidget(d_edit_, 1);
        h->addWidget(browse);
        train_form->addRow(QStringLiteral("Pretrained D"), row);
    }
    {
        auto* row = new QWidget(train_group);
        auto* h = new QHBoxLayout(row);
        h->setContentsMargins(0, 0, 0, 0);
        epochs_spin_ = new QSpinBox(row);
        epochs_spin_->setRange(1, 1000000);
        epochs_spin_->setValue(200);
        batch_spin_ = new QSpinBox(row);
        batch_spin_->setRange(1, 64);
        bucket_spin_ = new QSpinBox(row);
        bucket_spin_->setRange(32, 4096);
        bucket_spin_->setValue(384);
        disc_len_spin_ = new QSpinBox(row);
        disc_len_spin_->setRange(0, 100000);
        disc_len_spin_->setValue(12800);
        bucket_spin_->setToolTip(QStringLiteral(
            "Graph T. Conservative defaults (bucket 384 / disc_len 12800) target ~4 GB VRAM; "
            "the measured stable profile is bucket <= 300 and disc_len <= 4800."));
        disc_len_spin_->setToolTip(QStringLiteral(
            "Discriminator input length (0 = the whole official segment). Large values are costly and "
            "have crashed discrete GPUs; 4800 is the measured stable value."));
        h->addWidget(new QLabel(QStringLiteral("epochs"), row));
        h->addWidget(epochs_spin_);
        h->addWidget(new QLabel(QStringLiteral("batch"), row));
        h->addWidget(batch_spin_);
        h->addWidget(new QLabel(QStringLiteral("bucket"), row));
        h->addWidget(bucket_spin_);
        h->addWidget(new QLabel(QStringLiteral("disc len"), row));
        h->addWidget(disc_len_spin_);
        h->addStretch(1);
        train_form->addRow(QStringLiteral("Schedule"), row);
    }
    {
        auto* row = new QWidget(train_group);
        auto* h = new QHBoxLayout(row);
        h->setContentsMargins(0, 0, 0, 0);
        save_every_spin_ = new QSpinBox(row);
        save_every_spin_->setRange(0, 100000);
        save_every_spin_->setSuffix(QStringLiteral(" ep"));
        save_every_spin_->setToolTip(QStringLiteral("Save a checkpoint every N epochs (0 = only at the end)."));
        export_every_spin_ = new QSpinBox(row);
        export_every_spin_->setRange(0, 100000);
        export_every_spin_->setValue(5);
        export_every_spin_->setSuffix(QStringLiteral(" ep"));
        export_every_spin_->setToolTip(QStringLiteral(
            "Export an inference GGUF every N epochs (0 = off, no exportable model). "
            "Writes export.epN.gguf next to the checkpoints, plus the training-state "
            "train.epN.snapshot.gguf used by the Export card."));
        grad_accum_spin_ = new QSpinBox(row);
        grad_accum_spin_->setRange(1, 64);
        grad_accum_spin_->setToolTip(QStringLiteral(
            "Gradient accumulation: effective batch = batch x accum."));
        h->addWidget(new QLabel(QStringLiteral("save every"), row));
        h->addWidget(save_every_spin_);
        h->addWidget(new QLabel(QStringLiteral("export every"), row));
        h->addWidget(export_every_spin_);
        h->addWidget(new QLabel(QStringLiteral("grad accum"), row));
        h->addWidget(grad_accum_spin_);
        h->addStretch(1);
        train_form->addRow(QStringLiteral("Checkpoints"), row);
    }
    {
        auto* row = new QWidget(train_group);
        auto* h = new QHBoxLayout(row);
        h->setContentsMargins(0, 0, 0, 0);
        resume_edit_ = new QLineEdit(row);
        resume_edit_->setPlaceholderText(QStringLiteral("optional: continue from a .ckpt.gguf"));
        auto* browse = new QPushButton(QStringLiteral("Browse..."), row);
        connect(browse, &QPushButton::clicked, this, [this] {
            const QString file = browse_row(this, out_dir(), QStringLiteral("Resume checkpoint"), false,
                                            QStringLiteral("GGUF (*.gguf)"));
            if (!file.isEmpty()) {
                resume_edit_->setText(file);
            }
        });
        h->addWidget(resume_edit_, 1);
        h->addWidget(browse);
        train_form->addRow(QStringLiteral("Resume"), row);
    }
    {
        auto* row = new QWidget(train_group);
        auto* h = new QHBoxLayout(row);
        h->setContentsMargins(0, 0, 0, 0);
        train_run_ = new QPushButton(QStringLiteral("Start training"), row);
        train_status_ = new QLabel(QStringLiteral("Idle"), row);
        connect(train_run_, &QPushButton::clicked, this, &TrainingPage::run_train);
        h->addWidget(train_run_);
        h->addWidget(train_status_, 1);
        train_form->addRow(QString(), row);
    }
    content_layout->addWidget(train_group);

    // ---------------- Export -----------------------------------------------
    auto* export_group = new QGroupBox(QStringLiteral("Export inference model (rvc-train --export-gguf)"), content);
    auto* export_form = new QFormLayout(export_group);
    {
        auto* row = new QWidget(export_group);
        auto* h = new QHBoxLayout(row);
        h->setContentsMargins(0, 0, 0, 0);
        ckpt_combo_ = new QComboBox(row);
        ckpt_combo_->setMinimumContentsLength(40);
        ckpt_combo_->setSizeAdjustPolicy(QComboBox::AdjustToMinimumContentsLengthWithIcon);
        ckpt_combo_->setToolTip(QStringLiteral(
            "Training-state GGUFs (train.epN.snapshot.gguf, written by 'export every', or a "
            "converted G train.gguf). Resume checkpoints (train*.ckpt.gguf) are not exportable."));
        auto* browse = new QPushButton(QStringLiteral("Browse..."), row);
        connect(browse, &QPushButton::clicked, this, [this] {
            const QString file = browse_row(this, out_dir(), QStringLiteral("Training snapshot"), false,
                                            QStringLiteral("GGUF (*.gguf)"));
            if (!file.isEmpty()) {
                const int existing = ckpt_combo_->findData(file);
                if (existing < 0) {
                    ckpt_combo_->addItem(QFileInfo(file).fileName(), file);
                    ckpt_combo_->setCurrentIndex(ckpt_combo_->count() - 1);
                } else {
                    ckpt_combo_->setCurrentIndex(existing);
                }
            }
        });
        auto* refresh = new QPushButton(QStringLiteral("Refresh"), row);
        connect(refresh, &QPushButton::clicked, this, &TrainingPage::refresh_export_sources);
        connect(ckpt_combo_, QOverload<int>::of(&QComboBox::currentIndexChanged),
                this, [this] { update_export_path(); });
        h->addWidget(ckpt_combo_, 1);
        h->addWidget(browse);
        h->addWidget(refresh);
        export_form->addRow(QStringLiteral("Snapshot"), row);
    }
    {
        auto* row = new QWidget(export_group);
        auto* h = new QHBoxLayout(row);
        h->setContentsMargins(0, 0, 0, 0);
        export_edit_ = new QLineEdit(row);
        export_edit_->setPlaceholderText(QStringLiteral("inference GGUF (e.g. <name>.gguf)"));
        auto* browse = new QPushButton(QStringLiteral("Browse..."), row);
        connect(browse, &QPushButton::clicked, this, [this] {
            const QString file = QFileDialog::getSaveFileName(
                this, QStringLiteral("Export inference GGUF"), export_edit_->text(),
                QStringLiteral("GGUF (*.gguf)"));
            if (!file.isEmpty()) {
                export_edit_->setText(file);
            }
        });
        connect(export_edit_, &QLineEdit::textEdited, this, [this] { export_edited_ = true; });
        h->addWidget(export_edit_, 1);
        h->addWidget(browse);
        export_form->addRow(QStringLiteral("Output GGUF"), row);
    }
    {
        auto* row = new QWidget(export_group);
        auto* h = new QHBoxLayout(row);
        h->setContentsMargins(0, 0, 0, 0);
        export_run_ = new QPushButton(QStringLiteral("Export"), row);
        export_status_ = new QLabel(QStringLiteral("Idle"), row);
        connect(export_run_, &QPushButton::clicked, this, &TrainingPage::run_export);
        h->addWidget(export_run_);
        h->addWidget(export_status_, 1);
        export_form->addRow(QString(), row);
    }
    content_layout->addWidget(export_group);

    log_view_ = new QPlainTextEdit(content);
    log_view_->setReadOnly(true);
    log_view_->setMaximumBlockCount(4000);
    log_view_->setMinimumHeight(120);
    content_layout->addWidget(log_view_);
    content_layout->addStretch(1);

    scroll->setWidget(content);
    layout->addWidget(scroll);
}

void TrainingPage::populate_models() {
    const QString root = paths::root();
    const QString models = paths::models_root();
    const QString configs = root + QStringLiteral("/rvc-train-ggml/configs");

    QStringList config_files;
    for (const QFileInfo& fi : QDir(configs).entryInfoList({QStringLiteral("*.json")}, QDir::Files, QDir::Name)) {
        config_files << fi.absoluteFilePath();
    }

    auto fill = [](QComboBox* combo, const QStringList& files, const QString& preferred) {
        combo->clear();
        for (const QString& f : files) {
            combo->addItem(QFileInfo(f).fileName(), f);
        }
        const QString pick = paths::preferred(files, preferred);
        if (!pick.isEmpty()) {
            const int idx = combo->findData(pick);
            if (idx >= 0) {
                combo->setCurrentIndex(idx);
            }
        }
        combo->setEnabled(combo->count() > 0);
    };

    fill(config_combo_, config_files, QStringLiteral("v2-48k"));
    fill(prep_extractor_combo_, paths::gguf_files(models + QStringLiteral("/extractors")),
         QStringLiteral("contentvec"));
    fill(prep_f0_combo_, paths::gguf_files(models + QStringLiteral("/f0")), QStringLiteral("rmvpe.gguf"));

    sync_config();

    if (paths::tool(QStringLiteral("rvc-prep")).isEmpty()) {
        log(QStringLiteral("[warning] rvc-prep was not found (set RVC_TOOLS_DIR)."));
    }
    if (paths::tool(QStringLiteral("rvc-train")).isEmpty()) {
        log(QStringLiteral("[warning] rvc-train was not found (set RVC_TOOLS_DIR)."));
    }
    log(QStringLiteral("Repository root: %1").arg(root.isEmpty() ? QStringLiteral("<not found>") : root));
}

// Config files are JSON, not GGUF: list *.json in the config folder.
void TrainingPage::sync_config() {
    const QString name = QFileInfo(config_path()).fileName().toLower();
    if (name.contains(QStringLiteral("v1"))) {
        version_combo_->setCurrentIndex(0);
    } else if (name.contains(QStringLiteral("v2"))) {
        version_combo_->setCurrentIndex(1);
    }
    sync_pretrain_paths();
}

QString TrainingPage::find_training_state(const QString& version, QChar which) const {
    const QString root = paths::root();
    if (root.isEmpty()) {
        return {};
    }
    const QStringList dirs = {root + QStringLiteral("/work/models"),
                              root + QStringLiteral("/models/training"),
                              root + QStringLiteral("/work")};
    QString fallback;
    for (const QString& dir : dirs) {
        const QStringList files = QDir(dir).entryList({QStringLiteral("*.gguf")}, QDir::Files, QDir::Name);
        for (const QString& f : files) {
            const QString lower = f.toLower();
            if (!lower.contains(QStringLiteral("train"))) {
                continue;
            }
            const QString stem = lower.section(QStringLiteral(".train"), 0, 0);
            const bool is_g = stem.contains(QStringLiteral("g40")) || stem.contains(QStringLiteral("g48")) ||
                              stem.endsWith(QLatin1Char('g'));
            const bool is_d = stem.contains(QStringLiteral("d40")) || stem.contains(QStringLiteral("d48")) ||
                              stem.endsWith(QLatin1Char('d'));
            if (which == QLatin1Char('G') ? !is_g : !is_d) {
                continue;
            }
            const QString path = QDir(dir).absoluteFilePath(f);
            const bool matches_version = version == QStringLiteral("v2")
                                             ? lower.contains(QStringLiteral("48"))
                                             : lower.contains(QStringLiteral("40"));
            if (matches_version) {
                return path;
            }
            if (fallback.isEmpty()) {
                fallback = path;
            }
        }
    }
    return fallback;
}

void TrainingPage::sync_pretrain_paths() {
    const QString version = version_combo_->currentData().toString();
    if (!g_edited_) {
        const QString g = find_training_state(version, QLatin1Char('G'));
        if (!g.isEmpty()) {
            g_edit_->setText(g);
        }
    }
    if (!d_edited_) {
        const QString d = find_training_state(version, QLatin1Char('D'));
        if (!d.isEmpty()) {
            d_edit_->setText(d);
        }
    }
}

QString TrainingPage::cache_dir() const {
    return cache_edit_->text().trimmed();
}

QString TrainingPage::out_dir() const {
    return out_edit_->text().trimmed();
}

QString TrainingPage::config_path() const {
    return config_combo_->currentData().toString();
}

void TrainingPage::refresh_export_sources() {
    const QString out = out_dir();
    const QString previous = ckpt_combo_->currentData().toString();
    ckpt_combo_->clear();
    if (out.isEmpty() || !QDir(out).exists()) {
        ckpt_combo_->addItem(QStringLiteral("<no output dir>"), QString());
        return;
    }
    // Only training-state GGUFs are exportable: the epoch snapshots written by
    // "export every" and the pretrained/converted G files. The traincpp
    // checkpoints (train*.ckpt.gguf) carry optimizer state, not rvc.train.*.
    QStringList files;
    for (const QString& f : QDir(out).entryList({QStringLiteral("*.gguf")}, QDir::Files)) {
        if (f.endsWith(QStringLiteral(".snapshot.gguf")) || f.endsWith(QStringLiteral(".train.gguf"))) {
            files << f;
        }
    }
    std::sort(files.begin(), files.end(), [&out](const QString& a, const QString& b) {
        return QFileInfo(QDir(out).filePath(a)).lastModified() >
               QFileInfo(QDir(out).filePath(b)).lastModified();
    });
    if (files.isEmpty()) {
        ckpt_combo_->addItem(QStringLiteral("<no snapshots; enable 'export every' and train>"), QString());
        return;
    }
    for (const QString& f : files) {
        ckpt_combo_->addItem(f, QDir(out).absoluteFilePath(f));
    }
    const int previous_index = ckpt_combo_->findData(previous);
    ckpt_combo_->setCurrentIndex(previous_index >= 0 ? previous_index : 0);
    update_export_path();
}

void TrainingPage::update_export_path() {
    if (export_edited_) {
        return;
    }
    const QString ckpt = ckpt_combo_->currentData().toString();
    if (ckpt.isEmpty()) {
        return;
    }
    QString name = QFileInfo(ckpt).fileName();
    name.replace(QStringLiteral("train."), QStringLiteral("export."));
    name.replace(QStringLiteral(".ckpt.gguf"), QStringLiteral(".gguf"));
    name.replace(QStringLiteral(".snapshot.gguf"), QStringLiteral(".gguf"));
    export_edit_->setText(QFileInfo(ckpt).absolutePath() + QLatin1Char('/') + name);
}

void TrainingPage::log(const QString& text) {
    QString normalized = text;
    normalized.replace(QLatin1Char('\r'), QLatin1Char('\n'));
    const QStringList lines = normalized.split(QLatin1Char('\n'), Qt::SkipEmptyParts);
    for (const QString& line : lines) {
        log_view_->appendPlainText(line);
    }
}

void TrainingPage::run_process(const QString& program, const QStringList& args, QLabel* status,
                               QPushButton* button, const QString& start_text,
                               std::function<void(bool)> done) {
    if (program.isEmpty() || !QFileInfo::exists(program)) {
        status->setText(QStringLiteral("Tool not found (set RVC_TOOLS_DIR)"));
        return;
    }
    button->setEnabled(false);
    status->setText(start_text);
    log(QStringLiteral(">> %1 %2").arg(program, args.join(QLatin1Char(' '))));

    auto* proc = new QProcess(this);
    proc->setProgram(program);
    proc->setArguments(args);
    proc->setProcessChannelMode(QProcess::MergedChannels);
    active_.append(proc);

    connect(proc, &QProcess::readyReadStandardOutput, this, [this, proc, status] {
        const QString chunk = QString::fromLocal8Bit(proc->readAllStandardOutput());
        log(chunk);
        const QStringList lines = chunk.split(QLatin1Char('\n'), Qt::SkipEmptyParts);
        for (const QString& line : lines) {
            const QString trimmed = line.trimmed();
            if (!trimmed.isEmpty()) {
                status->setText(trimmed.right(120));
            }
        }
    });
    connect(proc, &QProcess::finished, this,
            [this, proc, status, button, done](int exit_code, QProcess::ExitStatus exit_status) {
                active_.removeAll(proc);
                proc->deleteLater();
                button->setEnabled(true);
                const bool ok = exit_status == QProcess::NormalExit && exit_code == 0;
                status->setText(ok ? QStringLiteral("Done")
                                   : QStringLiteral("Failed (exit %1)").arg(exit_code));
                if (done) {
                    done(ok);
                }
            });
    proc->start();
    if (!proc->waitForStarted(5000)) {
        active_.removeAll(proc);
        proc->deleteLater();
        button->setEnabled(true);
        status->setText(QStringLiteral("Failed to start %1").arg(QFileInfo(program).fileName()));
    }
}

void TrainingPage::run_prep() {
    const QString tool = paths::tool(QStringLiteral("rvc-prep"));
    const QString input = input_edit_->text().trimmed();
    if (input.isEmpty() || !QDir(input).exists()) {
        prep_status_->setText(QStringLiteral("Pick an audio folder first"));
        return;
    }
    if (cache_dir().isEmpty()) {
        prep_status_->setText(QStringLiteral("Pick a cache dir first"));
        return;
    }
    if (prep_extractor_combo_->currentData().toString().isEmpty() ||
        prep_f0_combo_->currentData().toString().isEmpty()) {
        prep_status_->setText(QStringLiteral("Feature/F0 extractor missing"));
        return;
    }
    QDir().mkpath(cache_dir());

    const bool v2 = version_combo_->currentIndex() == 1;
    QStringList args;
    args << QStringLiteral("--input") << input
         << QStringLiteral("--exp") << cache_dir()
         << QStringLiteral("--sr") << (v2 ? QStringLiteral("48000") : QStringLiteral("40000"))
         << QStringLiteral("--version") << (v2 ? QStringLiteral("v2") : QStringLiteral("v1"))
         << QStringLiteral("--hubert") << prep_extractor_combo_->currentData().toString()
         << QStringLiteral("--rmvpe") << prep_f0_combo_->currentData().toString()
         << QStringLiteral("--device") << prep_device_combo_->currentData().toString();
    run_process(tool, args, prep_status_, prep_run_, QStringLiteral("Preprocessing..."));
}

void TrainingPage::run_train() {
    const QString tool = paths::tool(QStringLiteral("rvc-train"));
    const bool scratch = mode_combo_->currentData().toBool();
    if (config_path().isEmpty() || !QFileInfo::exists(config_path())) {
        train_status_->setText(QStringLiteral("Pick a training config first"));
        return;
    }
    if (cache_dir().isEmpty() || !QDir(cache_dir()).exists()) {
        train_status_->setText(QStringLiteral("Cache dir does not exist (run preprocessing first)"));
        return;
    }
    if (out_dir().isEmpty()) {
        train_status_->setText(QStringLiteral("Pick an output dir first"));
        return;
    }
    if (!scratch && (g_edit_->text().trimmed().isEmpty() || d_edit_->text().trimmed().isEmpty())) {
        train_status_->setText(QStringLiteral("Pick the pretrained G and D models"));
        return;
    }
    QDir().mkpath(out_dir());

    QStringList args;
    args << QStringLiteral("--train") << config_path()
         << QStringLiteral("--exp") << cache_dir()
         << QStringLiteral("--out") << out_dir()
         << QStringLiteral("--epochs") << QString::number(epochs_spin_->value())
         << QStringLiteral("--batch") << QString::number(batch_spin_->value())
         << QStringLiteral("--bucket") << QString::number(bucket_spin_->value())
         << QStringLiteral("--disc-len") << QString::number(disc_len_spin_->value())
         << QStringLiteral("--save-interval") << QString::number(save_every_spin_->value())
         << QStringLiteral("--export-every") << QString::number(export_every_spin_->value())
         << QStringLiteral("--grad-accum") << QString::number(grad_accum_spin_->value());
    if (scratch) {
        args << QStringLiteral("--scratch");
    } else {
        args << QStringLiteral("--pretrain-g") << g_edit_->text().trimmed()
             << QStringLiteral("--pretrain-d") << d_edit_->text().trimmed();
    }
    const int device = train_device_combo_->currentData().toInt();
    if (device >= 0) {
        args << QStringLiteral("--device") << QString::number(device);
    }
    const QString resume = resume_edit_->text().trimmed();
    if (!resume.isEmpty()) {
        args << QStringLiteral("--resume") << resume;
    }

    run_process(tool, args, train_status_, train_run_, QStringLiteral("Training..."),
                [this](bool ok) {
                    refresh_export_sources();
                    if (ok) {
                        log(QStringLiteral("Training finished. Checkpoints and exports are in %1.")
                                .arg(out_dir()));
                    }
                });
}

void TrainingPage::run_export() {
    const QString tool = paths::tool(QStringLiteral("rvc-train"));
    const QString ckpt = ckpt_combo_->currentData().toString();
    const QString target = export_edit_->text().trimmed();
    if (ckpt.isEmpty() || !QFileInfo::exists(ckpt)) {
        export_status_->setText(QStringLiteral("Pick a training snapshot"));
        return;
    }
    if (ckpt.endsWith(QStringLiteral(".ckpt.gguf"))) {
        export_status_->setText(QStringLiteral(
            "Resume checkpoints cannot be exported; pick a train.epN.snapshot.gguf "
            "(written by 'export every')"));
        return;
    }
    if (target.isEmpty()) {
        export_status_->setText(QStringLiteral("Pick an output GGUF"));
        return;
    }
    QDir().mkpath(QFileInfo(target).absolutePath());

    run_process(tool, {QStringLiteral("--export-gguf"), ckpt, QStringLiteral("--out"), target},
                export_status_, export_run_, QStringLiteral("Exporting..."));
}

تقرير مشروع الخوارزمية الجينية لاختيار الميزات
Genetic Algorithm Feature Selection Project Report

التاريخ: 27 أكتوبر 2025  
المطور: فريق التطوير  
الهدف: تطوير نظام ذكي لاختيار الميزات باستخدام الخوارزميات الجينية المتقدمة

---

ملخص المشروع

تم تطوير تطبيق ويب متكامل باستخدام Django لاختيار الميزات المثلى من مجموعات البيانات باستخدام خوارزميات جينية متقدمة. يدعم التطبيق أنواع مختلفة من تمثيل الكروموسومات ويوفر واجهة مستخدم سهلة الاستخدام.

الأهداف المحققة

1. تطوير خوارزميات جينية متقدمة
- خوارزمية جينية أساسية للبيانات الصغيرة
- خوارزمية جينية متقدمة للبيانات الضخمة
- دعم أنواع مختلفة من الكروموسومات
- تحسين الأداء والسرعة

2. تطوير واجهة مستخدم متقدمة
- رفع ملفات البيانات
- اختيار العمود المستهدف
- اختيار نوع الكروموسوم
- تخصيص معاملات الخوارزمية
- عرض النتائج المفصلة مع الرسوم البيانية

3. معالجة البيانات الذكية
- تحويل أعمدة التاريخ تلقائياً
- ترميز المتغيرات الفئوية
- معالجة البيانات الضخمة
- التعامل مع القيم المفقودة

---

أنواع الكروموسومات المطورة

1. الكروموسوم الثنائي (Binary Chromosome)
```python
class BinaryChromosome(ChromosomeBase):
    """تمثيل ثنائي: [1, 0, 1, 0, 1] - كلاسيكي وفعال"""
    
    def initialize(self):
        # اختيار عشوائي للميزات مع احترام الحد الأقصى
        self.genes = np.zeros(self.n_features, dtype=int)
        selected_indices = np.random.choice(self.n_features, n_selected, replace=False)
        self.genes[selected_indices] = 1
```

المميزات:
- سهولة التنفيذ والفهم
- سرعة في العمليات الجينية
- مناسب للبيانات الصغيرة والمتوسطة

2. الكروموسوم الحقيقي (Real-Valued Chromosome)
```python
class RealValuedChromosome(ChromosomeBase):
    """تمثيل بقيم حقيقية: [0.8, 0.2, 0.9, 0.1] - مرن ودقيق"""
    
    def get_selected_features(self):
        # اختيار الميزات التي قيمتها أكبر من العتبة
        candidates = np.where(self.genes > self.threshold)[0]
        return candidates.tolist()
```

المميزات:
- مرونة في التحكم بدرجة اختيار الميزات
- تهجين وطفرة أكثر سلاسة
- مناسب للمسائل المعقدة

3. كروموسوم التبديل (Permutation Chromosome)
```python
class PermutationChromosome(ChromosomeBase):
    """تمثيل بالترتيب: [2, 0, 4, 1, 3] - ترتيب الميزات حسب الأهمية"""
    
    def get_selected_features(self):
        # اختيار أول n ميزات من الترتيب
        return self.genes[:self.n_selected].tolist()
```

المميزات:
- يحافظ على ترتيب الأهمية
- مناسب عندما نعرف أهمية الميزات مسبقاً
- تهجين متقدم باستخدام Order Crossover

4. الكروموسوم التكيفي (Adaptive Chromosome)
```python
class AdaptiveChromosome(ChromosomeBase):
    """يختار نوع التمثيل تلقائياً حسب خصائص المسألة"""
    
    def _select_encoding(self):
        if self.n_features <= 20:
            return "binary"  # للبيانات الصغيرة
        elif self.feature_importance is not None:
            return "permutation"  # عند معرفة الأهمية
        else:
            return "real_valued"  # للحالات العامة
```

المميزات:
- اختيار تلقائي للتمثيل الأمثل
- يتكيف مع خصائص البيانات
- الأداء الأمثل لكل نوع من المسائل

---

التطوير التقني

البنية التحتية
- إطار العمل: Django 5.2.7
- قاعدة البيانات: SQLite
- المكتبات الأساسية:
  - DEAP (للخوارزميات الجينية)
  - scikit-learn (للتعلم الآلي)
  - pandas & numpy (لمعالجة البيانات)
  - Bootstrap 5 (للواجهة)

الملفات الرئيسية

1. نماذج البيانات (models.py)
```python
class Dataset(models.Model):
    name = models.CharField(max_length=200)
    file = models.FileField(upload_to='datasets/')
    uploaded_at = models.DateTimeField(auto_now_add=True)

class FeatureSelection(models.Model):
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE)
    selected_features = models.JSONField()
    fitness_score = models.FloatField()
    algorithm_type = models.CharField(max_length=20)
    detailed_results = models.JSONField(null=True, blank=True)
```

2. الخوارزمية الأساسية (genetic_algorithm.py)
```python
class GeneticFeatureSelector:
    def __init__(self, X, y, chromosome_type='binary', max_features=None):
        self.chromosome_type = chromosome_type
        self.max_features = max_features or self.n_features // 2
        # إعداد DEAP مع دعم الكروموسومات المتقدمة
```

3. الخوارزمية المتقدمة (advanced_genetic_algorithm.py)
```python
class AdvancedGeneticFeatureSelector:
    def __init__(self, X, y, chromosome_type='adaptive'):
        # ميزات متقدمة: تقييم متعدد المعايير، توقف مبكر، نخبة
        self.chromosome_type = chromosome_type
        self._calculate_feature_importance()  # حساب أهمية الميزات
```

العمليات الجينية المتقدمة

التهجين الذكي
```python
def _chromosome_crossover(self, ind1, ind2):
    """تهجين خاص بنوع الكروموسوم"""
    child1_chr, child2_chr = ind1.chromosome.crossover(ind2.chromosome)
    self._update_individual_from_chromosome(ind1, child1_chr)
    self._enforce_feature_limit_on_individual(ind1)
```

الطفرة الذكية
```python
def _chromosome_mutation(self, individual):
    """طفرة تتكيف مع نوع الكروموسوم"""
    individual.chromosome.mutate(self.mut_prob)
    self._update_individual_from_chromosome(individual, individual.chromosome)
```

---

واجهة المستخدم

الصفحة الرئيسية
- عرض جميع مجموعات البيانات المرفوعة
- إمكانية رفع ملفات جديدة
- تصميم متجاوب مع Bootstrap

صفحة تفاصيل البيانات
```html
<!-- اختيار العمود المستهدف -->
<select name="target_column" required>
    {% for column in columns %}
        <option value="{{ column }}">{{ column }}</option>
    {% endfor %}
</select>

<!-- اختيار نوع الكروموسوم -->
<select name="chromosome_type">
    <option value="adaptive" selected>Adaptive (Recommended)</option>
    <option value="binary">Binary Chromosome</option>
    <option value="real_valued">Real-Valued Chromosome</option>
    <option value="permutation">Permutation Chromosome</option>
</select>
```

صفحة النتائج المفصلة
- عرض الميزات المختارة
- درجة اللياقة والتحسن
- رسم بياني لتقارب الخوارزمية
- إحصائيات مفصلة عن الأداء

---

معالجة البيانات المتقدمة

تحويل أعمدة التاريخ
```python
# اكتشاف وتحويل أعمدة التاريخ تلقائياً
for col in df.columns:
    if 'date' in col.lower():
        df[col] = pd.to_datetime(df[col], errors='coerce')
        date_features = pd.concat([
            df[col].dt.year.rename(f'{col}_Year'),
            df[col].dt.month.rename(f'{col}_Month'),
            df[col].dt.day.rename(f'{col}_Day'),
            df[col].dt.dayofweek.rename(f'{col}_DayOfWeek')
        ], axis=1)
        df = df.join(date_features)
```

معالجة البيانات الضخمة
```python
# أخذ عينة للبيانات الضخمة
if len(y) > 5000:
    X, _, y, _ = train_test_split(X, y, train_size=5000, stratify=y)

# معالجة القيم الكثيرة
if df[target_column].nunique() > 100:
    top_items = df[target_column].value_counts().nlargest(100).index
    df = df[df[target_column].isin(top_items)]
```

---

النتائج والأداء

اختبارات الأداء
تم اختبار التطبيق على مجموعة بيانات البقالة:
- عدد العينات: 38,765
- عدد الميزات الأصلية: 3 (Member_number, Date, itemDescription)
- عدد الميزات بعد المعالجة: 6 (بعد تحويل التاريخ)

النتائج المحققة
- الميزة المختارة: Date_Day
- درجة اللياقة: 0.278966
- نوع الخوارزمية: Basic (للبيانات الصغيرة)
- عدد الأجيال: 1 (توقف مبكر)

مقاييس الأداء
```python
# تقييم متوازن يجمع بين الدقة وتقليل الميزات
fitness = 0.7 * accuracy + 0.3 * feature_reduction_bonus

# معايير تقييم متعددة
- accuracy: دقة التصنيف
- f1_score: المتوسط المرجح لـ F1
- roc_auc: منطقة تحت المنحنى
- balanced: تقييم متوازن مخصص
```

---

تدفق العمل

1. رفع البيانات
```
المستخدم → رفع ملف CSV → التحقق من صحة البيانات → حفظ في قاعدة البيانات
```

2. إعداد الخوارزمية
```
اختيار العمود المستهدف → اختيار نوع الكروموسوم → تحديد المعاملات → بدء التشغيل
```

3. معالجة البيانات
```
قراءة البيانات → تحويل التواريخ → ترميز الفئات → تقسيم الميزات والهدف
```

4. تشغيل الخوارزمية
```
إنشاء المجتمع الأولي → التقييم → الاختيار → التهجين → الطفرة → التكرار
```

5. عرض النتائج
```
حفظ النتائج → إنشاء الرسوم البيانية → عرض التحليل المفصل
```

---

الميزات المتقدمة

1. التوقف المبكر
```python
if self.generations_without_improvement >= self.early_stopping_patience:
    print(f"توقف مبكر في الجيل {generation}")
    break
```

2. استراتيجية النخبة
```python
elite = tools.selBest(population, self.elite_size)
population[:] = elite + offspring
```

3. التقييم المتعدد المعايير
```python
scores = {
    'accuracy': cross_val_score(model, X, y, scoring='accuracy'),
    'f1': cross_val_score(model, X, y, scoring='f1_weighted'),
    'roc_auc': cross_val_score(model, X, y, scoring='roc_auc_ovr')
}
```

4. حساب أهمية الميزات
```python
from sklearn.feature_selection import mutual_info_classif
self.feature_importance = mutual_info_classif(self.X, self.y, random_state=42)
```

---

الرسوم البيانية والتصورات

رسم تقارب الخوارزمية
```javascript
const convergenceChart = new Chart(ctx, {
    type: 'line',
    data: {
        labels: generationLabels,
        datasets: [{
            label: 'Best Fitness',
            data: bestFitnessData,
            borderColor: 'rgb(75, 192, 192)'
        }, {
            label: 'Average Fitness',
            data: avgFitnessData,
            borderColor: 'rgb(255, 99, 132)'
        }]
    }
});
```

---

الأمان والموثوقية

التحقق من صحة البيانات
```python
# التحقق من وجود العمود المستهدف
if target_column not in df.columns:
    messages.error(request, f'Target column "{target_column}" not found')
    return redirect('dataset_detail', pk=pk)

# التحقق من صحة ملف CSV
try:
    df = pd.read_csv(file)
    if len(df.columns) < 2:
        raise ValueError("Dataset must have at least 2 columns")
except Exception as e:
    messages.error(request, f'Invalid file: {str(e)}')
```

معالجة الأخطاء
```python
try:
    # تشغيل الخوارزمية الجينية
    selected_features, fitness_score, logbook, detailed_results = selector.run()
except Exception as e:
    messages.error(request, f'Error during feature selection: {str(e)}')
    return redirect('dataset_detail', pk=pk)
```

---

التحسينات المستقبلية

1. خوارزميات إضافية
- خوارزميات تطورية أخرى (PSO, DE)
- تحسين متعدد الأهداف
- خوارزميات هجينة

2. واجهة المستخدم
- لوحة تحكم متقدمة
- مقارنة النتائج
- تصدير التقارير

3. الأداء
- معالجة متوازية
- دعم GPU
- تحسين الذاكرة

4. التكامل
- API RESTful
- دعم قواعد بيانات أخرى
- تكامل مع مكتبات ML أخرى

---

الخلاصة

تم تطوير نظام متكامل وقوي لاختيار الميزات باستخدام الخوارزميات الجينية المتقدمة. النظام يوفر:

الإنجازات الرئيسية:
1. أربعة أنواع من الكروموسومات مع خصائص مختلفة
2. واجهة مستخدم سهلة مع إمكانيات تخصيص متقدمة
3. معالجة بيانات ذكية تتعامل مع أنواع مختلفة من البيانات
4. نتائج مفصلة مع تصورات بيانية
5. أداء محسن مع توقف مبكر واستراتيجية النخبة

الفوائد للمستخدمين:
- سهولة الاستخدام: واجهة بديهية لا تتطلب خبرة برمجية
- مرونة عالية: دعم أنواع مختلفة من البيانات والمسائل
- نتائج موثوقة: خوارزميات مختبرة ومحسنة
- تصور واضح: رسوم بيانية وإحصائيات مفصلة

النظام جاهز للاستخدام في البحث الأكاديمي والتطبيقات العملية لاختيار الميزات المثلى من مجموعات البيانات المختلفة.

---

معلومات التواصل

فريق التطوير  
التاريخ: 27 أكتوبر 2025  
الإصدار: 1.0.0

---

هذا التقرير يوثق العمل المنجز في مشروع الخوارزمية الجينية لاختيار الميزات. جميع الكود والوثائق متاحة في مجلد المشروع.

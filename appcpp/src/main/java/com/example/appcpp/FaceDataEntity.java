import androidx.room.Entity;
import androidx.room.PrimaryKey;
import java.util.List;

@Entity(tableName = "face_data_table")
public class FaceDataEntity {

    @PrimaryKey(autoGenerate = true)
    public int id;

    public String name;
    public List<Float> embedding;

    public FaceDataEntity(String name, List<Float> embedding) {
        this.name = name;
        this.embedding = embedding;
    }
}

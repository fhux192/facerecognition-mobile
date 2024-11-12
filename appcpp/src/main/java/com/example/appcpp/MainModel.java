package com.example.appcpp;

import java.util.List;

public class MainModel {
    private String name;
    private List<Float> embedding; // Đảm bảo kiểu dữ liệu là List<Float>

    public MainModel() {} // Constructor mặc định cần thiết cho Firebase

    public MainModel(String name, List<Float> embedding) {
        this.name = name;
        this.embedding = embedding;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public List<Float> getEmbedding() {
        return embedding;
    }

    public void setEmbedding(List<Float> embedding) {
        this.embedding = embedding;
    }
}
